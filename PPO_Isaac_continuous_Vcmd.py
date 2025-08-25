# 导入自制的库
from utils.NN import *
from utils.Buffer import *
from utils.Useful_Function import *
from utils.Reward_Function import *
from utils.Isaac_Sim_Initialization import *

# 导入Isaac Sim
from isaacsim import SimulationApp

headless = 1  # 是否开启UI
sim = SimulationApp({"headless": headless})  # 启动Isaac Sim 软件， 必须放在导入Isaac sim 库之前。
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api.robots import RobotView
from isaacsim.core.prims import RigidPrim
from isaacsim.core.prims import Articulation
from omni.isaac.cloner import Cloner

#设置GPU显示
from isaacsim.core.utils.carb import set_carb_setting
import carb

settings = carb.settings.get_settings()
set_carb_setting(settings, "/persistent/omnihydra/useSceneGraphInstancing", True)


class PPO:
    def __init__(self, bot_cfg, train):
        # 设置 GPU 管线 （Pipline）
        self.device = torch.device("cuda:0")
        device = 'cuda'
        backend = "torch"
        print("Running on the GPU")

        # 导入配置文件
        self.cfg = bot_cfg

        # Robot Parameters initializations
        self.Kp = bot_cfg.RobotParam.Kp
        self.Kd = bot_cfg.RobotParam.Kd
        self.initial_height = bot_cfg.RobotParam.initial_height
        self.initial_body_vel_range = bot_cfg.RobotParam.initial_body_vel_range
        self.initial_joint_pos_range = bot_cfg.RobotParam.initial_joint_pos_range
        self.initial_joint_vel_range = bot_cfg.RobotParam.initial_joint_vel_range
        self.action_scale = bot_cfg.RobotParam.action_scale
        self.std_scale = bot_cfg.RobotParam.std_scale

        # Critic Initialization
        self.state_dim = bot_cfg.CriticParam.state_dim
        self.privilege_dim = bot_cfg.CriticParam.privilege_dim
        self.critic_layers_num = bot_cfg.CriticParam.critic_layers_num
        self.critic_lr = bot_cfg.CriticParam.critic_lr
        self.critic_update_frequency = bot_cfg.CriticParam.critic_update_frequency
        self.critic = Critic((self.state_dim + self.privilege_dim), self.critic_layers_num)

        # Actor Initialization
        self.act_layers_num = bot_cfg.ActorParam.act_layers_num
        self.actuator_num = bot_cfg.ActorParam.actuator_num
        self.actor_lr = bot_cfg.ActorParam.actor_lr
        self.actor_update_frequency = bot_cfg.ActorParam.actor_update_frequency
        self.actor = Actor(self.state_dim, self.act_layers_num, self.actuator_num, self.action_scale,
                           self.std_scale)

        # Training parameters
        self.train = train
        self.gamma = bot_cfg.PPOParam.gamma
        self.lam = bot_cfg.PPOParam.lam
        self.epsilon = bot_cfg.PPOParam.epsilon
        self.maximum_step = bot_cfg.PPOParam.maximum_step
        self.episode = bot_cfg.PPOParam.episode
        self.entropy_coef = bot_cfg.PPOParam.entropy_coef
        self.batch_size = bot_cfg.PPOParam.batch_size
        self.critic.to(self.device)
        self.actor.to(self.device)
        self.max_reward_sum = -90000111

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Loss functions
        self.loss_fn = torch.nn.MSELoss()

        # World initialization
        self.dt = bot_cfg.EnvParam.dt
        self.sub_step = bot_cfg.EnvParam.sub_step
        self.prim_path = bot_cfg.EnvParam.prim_path
        self.file_path = bot_cfg.EnvParam.file_path
        self.friction_coef = bot_cfg.EnvParam.friction_coef
        self.world = setup_world_physics(World, device, backend, self.dt, self.sub_step)
        self.agents_num = (bot_cfg.EnvParam.agents_num - bot_cfg.EnvParam.agents_num_in_play) * self.train + \
                          bot_cfg.EnvParam.agents_num_in_play

        # Mission Objective
        self.vel_cmd_scale = bot_cfg.MissionObjectiveParam.target_vel_scale
        self.target_height = bot_cfg.MissionObjectiveParam.target_height

        # 创建经验缓冲区
        self.Global_Experience_buffer = Agent_State_Buffer(
            self.state_dim, self.privilege_dim, self.actuator_num, self.agents_num,
            self.maximum_step, self.device)

    def scene_setup(self):
        self.initial_pos, self.initial_ori = create_square_positions(
            num_agents=self.agents_num, spacing=1, initial_height=self.initial_height)
        self.vel_cmd = torch.zeros(size=(self.agents_num, 3), device=self.device)
        self.target_ori = FT(np.array([[0, 0, 0]]))  # FT(np.array([0, 0, 0]))
        self.time = torch.zeros(size=(self.agents_num, 1), device=self.device)
        # 添加地面
        ground_plane_path = '/World/ground_plane'
        self.world.scene.add_default_ground_plane(
            prim_path=ground_plane_path,
            static_friction=self.friction_coef,
            dynamic_friction=self.friction_coef)

        # 设置碰撞过滤路径, 让机器人只和地面碰撞，机器人之间无碰撞
        collision_filter_paths = [ground_plane_path]
        world_prim_path = "/World/" + self.prim_path

        # 添加参考模型并克隆
        stage_utils.add_reference_to_stage(self.file_path, world_prim_path + "_0")
        cloner = Cloner()
        print("clone")
        target_paths = cloner.generate_paths(world_prim_path, self.agents_num)
        cloner.clone(source_prim_path=world_prim_path + "_0", prim_paths=target_paths)
        print("clone_complete")
        # 设置碰撞过滤
        cloner.filter_collisions(
            physicsscene_path=self.world.get_physics_context().prim_path,
            collision_root_path="/World/collisions",
            prim_paths=target_paths,
            global_paths=collision_filter_paths)

        # 把3D模型套到Articulation类之中，这样可以得到并施加机器人的属性
        prim_expr = world_prim_path + ".*"
        prim = Articulation(prim_paths_expr=prim_expr, name="rigid_prim_view")

        self.foot_L_sensor = RigidPrim(
            prim_paths_expr=prim_expr + "/" + self.prim_path + "/ankle_L_Link",
            name="RigidContactView1",
            max_contact_count=10,
            contact_filter_prim_paths_expr=collision_filter_paths)

        self.foot_R_sensor = RigidPrim(
            prim_paths_expr=prim_expr + "/" + self.prim_path + "/ankle_R_Link",
            name="RigidContactView2",
            max_contact_count=10,
            contact_filter_prim_paths_expr=collision_filter_paths)

        self.abad_L_Link = RigidPrim(
            prim_paths_expr=prim_expr + "/" + self.prim_path + "/abad_L_Link",
            name="RigidContactView3",
            max_contact_count=10,
            contact_filter_prim_paths_expr=collision_filter_paths)

        self.abad_R_Link = RigidPrim(
            prim_paths_expr=prim_expr + "/" + self.prim_path + "/abad_R_Link",
            name="RigidContactView4",
            max_contact_count=10,
            contact_filter_prim_paths_expr=collision_filter_paths)
        print("add prim")
        self.world.scene.add(prim)

        # 初始化世界和传感器
        print("world reset")
        self.world.reset()
        print("sensor reset")
        self.foot_L_sensor.initialize()
        self.foot_R_sensor.initialize()
        self.abad_L_Link.initialize()
        self.abad_R_Link.initialize()

    def prim_initialization(self, agent_index):
        """
        :param agent_index:  哪个序号的机器人挂了
        :return:
        """
        num_agents = len(agent_index)
        if num_agents == 0:
            return
        # 生成随机初始化数据
        initial_v = self.initial_body_vel_range * (2 * torch.rand(num_agents, 6, device=self.device) - 1)
        initial_joint_pos = self.initial_joint_pos_range * (2 * torch.rand(num_agents, 8, device=self.device) - 1)
        initial_joint_vel = self.initial_joint_vel_range * (2 * torch.rand(num_agents, 8, device=self.device) - 1)

        # 设置速度命令和时间
        self.vel_cmd[agent_index, 0] = self.vel_cmd_scale * torch.rand(num_agents, device=self.device)
        self.time[agent_index, 0] = 0

        # 获取prim并设置身体速度
        prim = self.world.scene.get_object("rigid_prim_view")
        prim.set_world_poses(positions=self.initial_pos[agent_index],
                             orientations=self.initial_ori[agent_index],
                             indices=agent_index)
        prim.set_velocities(initial_v, indices=agent_index)
        prim.set_joint_velocities(initial_joint_vel, indices=agent_index)
        prim.set_joint_positions(initial_joint_pos, indices=agent_index)

        if not self.train:  #当你play的时候，设置速度指令
            self.vel_cmd[:, 0] = -1

    def get_store_observation(self, current_step):
        # 获取机器人
        prim = self.world.scene.get_object("rigid_prim_view")
        # #——————————————————————获取当前时刻状态————————————————————————————————##
        self.body_pos, self.body_ori = prim.get_world_poses()
        self.body_ori = get_euler_angle(self.body_ori)
        self.joint_pos = prim.get_joint_positions()
        self.joint_vel = prim.get_joint_velocities()

        # 获取时间和时钟信号
        clock_signal = 2 * torch.pi * self.time
        self.sine_clock = torch.sin(clock_signal)
        self.cosine_clock = torch.cos(clock_signal)

        # 拼接出下一时刻状态空间张量，并归一化
        current_state = torch.concatenate(
            (self.joint_pos,
             self.joint_vel,
             self.body_ori,
             self.vel_cmd,
             self.sine_clock,
             self.cosine_clock
             ), dim=1)

        normalized_state = State_Normalization(current_state)
        # #——————————————————————获取当前时刻状态结束————————————————————————————————##

        # #——————————————————————获取额外机器人状态————————————————————————————————##
        # 获得机器人body高度 和 Abad 关节角度
        self.linear_vel = prim.get_linear_velocities()
        self.body_height = self.body_pos[:, 2]
        self.abad_L_R_Joint_pos = self.joint_pos[:, :2]
        L_foot_contact_force = self.foot_L_sensor.get_net_contact_forces()
        R_foot_contact_force = self.foot_R_sensor.get_net_contact_forces()
        L_foot_contact_situation = torch.any(L_foot_contact_force > 1e-5, dim=1).view(-1, 1).float()
        R_foot_contact_situation = torch.any(R_foot_contact_force > 1e-5, dim=1).view(-1, 1).float()
        self.L_foot_ori = get_euler_angle(self.foot_L_sensor.get_world_poses()[1])
        self.R_foot_ori = get_euler_angle(self.foot_R_sensor.get_world_poses()[1])
        self.L_foot_z = self.foot_L_sensor.get_world_poses()[0][:, 2].view(-1, 1)
        self.R_foot_z = self.foot_R_sensor.get_world_poses()[0][:, 2].view(-1, 1)
        self.L_foot_z_dot = self.foot_L_sensor.get_linear_velocities()[:, 2].view(-1, 1)
        self.R_foot_z_dot = self.foot_R_sensor.get_linear_velocities()[:, 2].view(-1, 1)
        # #——————————————————————获取额外机器人状态结束————————————————————————————————##
        current_privilege = torch.concatenate(
            (self.body_height.view(-1, 1),
             self.linear_vel,
             L_foot_contact_situation,
             R_foot_contact_situation
             ), dim=1)
        # #——————————————————————储存状态————————————————————————————————##
        self.Global_Experience_buffer.store_state(normalized_state, current_step)
        self.Global_Experience_buffer.store_privilege_state(current_privilege, current_step)
        # #——————————————————————储存状态结束————————————————————————————————##
        return normalized_state

    def get_store_action(self, normalized_state, current_step):
        self.effort = 0  # 用于计算关节扭矩使用量
        with torch.no_grad():
            mu, std = self.actor(normalized_state)
            self.std_sum += std.mean()
            action = torch.normal(mu, std).clip(-self.action_scale, self.action_scale) if self.train else mu

            # 将动作储存到buffer中
            self.Global_Experience_buffer.store_action(action, current_step)

            return action

    def apply_PD_target(self, angle):
        prim = self.world.scene.get_object("rigid_prim_view")

        # 获取机器人关节信息
        joint_pos = prim.get_joint_positions()
        joint_vel = prim.get_joint_velocities()

        # 计算PD关机扭矩，对脚踝的PD参数调整
        torque = (angle - joint_pos) * self.Kp - joint_vel * self.Kd
        torque[:, :6] = torque[:, :6].clip(-80, 80)  #设置关节最大扭矩
        torque[:, -2:] = torque[:, -2:].clip(-20, 20)

        # 累加扭矩用量，用于奖励计算
        self.effort += torch.sum(torque ** 2, dim=1) / (self.sub_step * 8)

        # 施加扭矩
        prim.set_joint_efforts(torque)

    def get_store_state_reward_over(self, current_step):
        prim = self.world.scene.get_object("rigid_prim_view")
        # #——————————————————————获取下一时刻状态————————————————————————————————##
        # 获取机器人的关节身体信息
        self.next_body_pos, self.next_body_ori = prim.get_world_poses()
        self.next_body_ori = get_euler_angle(self.next_body_ori)
        self.next_joint_pos = prim.get_joint_positions()
        self.next_joint_vel = prim.get_joint_velocities()

        # 获取时间和时钟信号
        self.time += self.dt
        clock_signal = 2 * torch.pi * self.time
        self.sine_clock = torch.sin(clock_signal)
        self.cosine_clock = torch.cos(clock_signal)

        # 拼接出下一时刻状态空间张量，并归一化
        next_state = torch.concatenate((self.next_joint_pos,
                                        self.next_joint_vel,
                                        self.next_body_ori,
                                        self.vel_cmd,
                                        self.sine_clock,
                                        self.cosine_clock), dim=1)
        normalized_next_state = State_Normalization(next_state)
        # #——————————————————————获取下一时刻状态结束————————————————————————————————##

        # #——————————————————————获取额外机器人状态————————————————————————————————##
        # 获得机器人body高度 和 Abad 关节角度
        self.next_body_height = self.next_body_pos[:, 2]
        self.next_abad_L_R_Joint_pos = self.next_joint_pos[:, :2]
        self.next_ankle_L_R_Joint_pos = self.next_joint_pos[:, -2:]
        self.next_linear_vel = prim.get_linear_velocities()
        L_foot_contact_force = self.foot_L_sensor.get_net_contact_forces()
        R_foot_contact_force = self.foot_R_sensor.get_net_contact_forces()
        L_foot_contact_situation = torch.any(L_foot_contact_force > 1e-5, dim=1).view(-1, 1).float()
        R_foot_contact_situation = torch.any(R_foot_contact_force > 1e-5, dim=1).view(-1, 1).float()

        self.next_L_foot_ori = get_euler_angle(self.foot_L_sensor.get_world_poses()[1])
        self.next_R_foot_ori = get_euler_angle(self.foot_R_sensor.get_world_poses()[1])

        self.next_L_foot_z = self.foot_L_sensor.get_world_poses()[0][:, 2].view(-1, 1)
        self.next_R_foot_z = self.foot_R_sensor.get_world_poses()[0][:, 2].view(-1, 1)
        self.next_L_foot_z_dot = self.foot_L_sensor.get_linear_velocities()[:, 2].view(-1, 1)
        self.next_R_foot_z_dot = self.foot_R_sensor.get_linear_velocities()[:, 2].view(-1, 1)

        des_L_foot_pos, des_R_foot_pos, des_L_dot, des_R_dot = foot_pos_generate(clock_signal)

        # #——————————————————————获取额外机器人状态结束————————————————————————————————##
        next_privilege = torch.concatenate(
            (self.next_body_height.view(-1, 1),
             self.next_linear_vel,
             L_foot_contact_situation,
             R_foot_contact_situation
             ), dim=1)

        # ——————————————————————奖励————————————————————————————————
        # 速度跟踪
        reward_vel_track = 8 * potential_reward(self.vel_cmd[:, :2], self.linear_vel[:, :2],
                                                self.next_linear_vel[:, :2],
                                                gamma=1)

        # 高度跟踪
        reward_height_track = 8 * potential_reward(self.target_height, self.body_height, self.next_body_height,
                                                   gamma=1)

        # 方向跟踪
        reward_ori_track = 8 * potential_reward(self.target_ori[:, 2],
                                                self.body_ori[:, 2],
                                                self.next_body_ori[:, 2], gamma=1)
        reward_ori_track += 2 * potential_reward(self.target_ori,
                                                self.body_ori,
                                                self.next_body_ori, gamma=1)

        reward_ori_track += 0.1 * potential_reward(self.target_ori[:, :2],
                                                 self.L_foot_ori[:, :2],
                                                 self.next_L_foot_ori[:, :2], gamma=1)

        reward_ori_track += 0.1 * potential_reward(self.target_ori[:, :2],
                                                 self.R_foot_ori[:, :2],
                                                 self.next_R_foot_ori[:, :2], gamma=1)

        # 关节角度正则化
        joint_regularization_reward = 3 * potential_reward(0,
                                                           self.abad_L_R_Joint_pos,
                                                           self.next_abad_L_R_Joint_pos,
                                                           gamma=1)

        # 奖励SF
        walking_phase_reward = 0.4 * walking_phase(self.sine_clock,
                                                   self.cosine_clock,
                                                   L_foot_contact_force,
                                                   R_foot_contact_force)

        foot_pos_track_reward = 0 * (abs_sum(des_L_foot_pos, self.next_L_foot_z)
                                     + abs_sum(des_R_foot_pos, self.next_R_foot_z))

        foot_vel_track_reward = 0 * (potential_reward(des_L_dot, self.L_foot_z_dot, self.next_L_foot_z_dot)
                                     + potential_reward(des_R_dot, self.R_foot_z_dot, self.next_R_foot_z_dot))
        # 惩罚扭矩使用
        joint_effort_reward = -0.0000 * self.effort

        # 惩罚失败
        over1 = torch.any(torch.abs(self.next_body_ori) > np.pi / 4, dim=1)
        over2 = (self.next_body_height < 0.7)
        over3 = torch.any(torch.abs(self.next_abad_L_R_Joint_pos) > np.pi / 6, dim=1)
        over4 = torch.abs(self.next_body_ori[:, 2]) > np.pi / 6

        over = over1 | over2 | over3 | over4  # Element-wise OR
        reward_fall = - over.float() * 20
        reward = reward_fall + \
                 reward_ori_track + \
                 reward_height_track + \
                 reward_vel_track + \
                 walking_phase_reward + \
                 +foot_pos_track_reward + \
                 +foot_vel_track_reward + \
                 + joint_regularization_reward + joint_effort_reward
        # ——————————————————————奖励结束————————————————————————————————#

        # ——————————————————————计算平均奖励————————————————————————————————#
        self.reward_sum += reward.mean()
        self.vel_track_sum += reward_vel_track.mean()
        self.height_track_sum += reward_height_track.mean()
        self.ori_track_sum += reward_ori_track.mean()
        self.falling_sum += reward_fall.mean()
        self.walking_phase_reward_sum += walking_phase_reward.float().mean()
        self.foot_pos_track_reward_sum += foot_pos_track_reward.mean()
        self.foot_vel_track_reward_sum += foot_vel_track_reward.mean()

        self.joint_regularization_reward_sum += joint_regularization_reward.mean()
        self.joint_effort_reward_sum += joint_effort_reward.mean()

        # ——————————————————————计算平均奖励结束————————————————————————————————#

        # ——————————————————————PPO储存rt,st+1,over————————————————————————————————#
        self.Global_Experience_buffer.store_next_state(normalized_next_state, current_step)
        self.Global_Experience_buffer.store_next_privilege_state(next_privilege, current_step)
        self.Global_Experience_buffer.store_reward(reward.view(-1, 1), current_step)
        self.Global_Experience_buffer.store_over(over.view(-1, 1), current_step)
        # ——————————————————————PPO储存rt,st+1,over结束————————————————————————————————#

        # ——————————————————————把over的机器人位置初始化————————————————————————————————#
        self.prim_initialization(torch.nonzero(over).flatten())
        # ——————————————————————把over的机器人位置初始化结束————————————————————————————————#

    def play(self):

        self.scene_setup()
        self.prim_initialization(torch.arange(0, self.agents_num))
        print("Setup Complete")
        for epi in range(self.episode):
            print(f"---------------------the {epi} episode started------------------------")
            self.reward_sum = 0
            self.vel_track_sum = 0
            self.height_track_sum = 0
            self.ori_track_sum = 0
            self.falling_sum = 0
            self.std_sum = 0
            self.walking_phase_reward_sum = 0
            self.foot_pos_track_reward_sum = 0
            self.foot_vel_track_reward_sum = 0
            self.joint_regularization_reward_sum = 0
            self.joint_effort_reward_sum = 0
            for current_step in range(self.maximum_step):
                # 观测 St, 存St
                normalized_state = self.get_store_observation(current_step)

                # 做 At， 存At
                angle = self.get_store_action(normalized_state, current_step)

                # 前往状态 St+1
                for _ in range(self.sub_step):
                    self.apply_PD_target(angle)
                    self.world.step(render=False)
                self.apply_PD_target(angle)
                self.world.step(render=not self.train)

                # 计算reward，over。 完成存储
                self.get_store_state_reward_over(current_step)

            if self.train:
                self.update()  # 更新
                self.save_each_epi_model()  # 每回合保存神经网络文件
                print("reward_sum:", self.reward_sum.item() / self.maximum_step)
                print("vel_track_reward:", self.vel_track_sum.item() / self.maximum_step)
                print("ori_track_reward:", self.ori_track_sum.item() / self.maximum_step)
                print("height_track_reward:", self.height_track_sum.item() / self.maximum_step)
                print("falling_reward:", self.falling_sum.item() / self.maximum_step)
                print("contact_reward_sum:", self.walking_phase_reward_sum.item() / self.maximum_step)
                print("foot_pos_track_reward_sum:", self.foot_pos_track_reward_sum.item() / self.maximum_step)
                print("foot_vel_track_reward_sum:", self.foot_vel_track_reward_sum.item() / self.maximum_step)
                print("std_sum:", self.std_sum.item() / self.maximum_step)
                print("joint_regularization:", self.joint_regularization_reward_sum.item() / self.maximum_step)
                print("joint_effort_reward_sum", self.joint_effort_reward_sum.item() / self.maximum_step)

                if self.reward_sum > self.max_reward_sum:  # 保存最高分的神经网络文件
                    self.save_model()
                    self.max_reward_sum = self.reward_sum
                    print(f"upgrade!!!!!")

    def update(self):
        # 获取经验数据
        buffer = self.Global_Experience_buffer
        state = buffer.state_buffer.view(-1, self.state_dim)
        privilege = buffer.privilege_buffer.view(-1, self.state_dim + self.privilege_dim)
        action = buffer.action_buffer.view(-1, self.actuator_num)
        next_state = buffer.next_state_buffer.view(-1, self.state_dim)
        next_privilege = buffer.next_privilege_buffer.view(-1, self.state_dim + self.privilege_dim)
        reward = buffer.reward_buffer.view(-1, 1)
        over = buffer.over_buffer.view(-1, 1)

        # 计算旧策略概率
        with torch.no_grad():
            mu_old, std_old = self.actor(state)
            old_prob = torch.distributions.Normal(mu_old, std_old).log_prob(action).sum(dim=1, keepdim=True).exp()

        # Critic更新
        for _ in range(self.critic_update_frequency):
            idx = np.random.choice(len(state), self.batch_size, replace=False)
            p_batch, np_batch, r_batch, o_batch = privilege[idx], next_privilege[idx], reward[idx], over[idx]

            value = self.critic(p_batch)
            with torch.no_grad():
                next_value = self.critic(np_batch)
                target_value = r_batch + self.gamma * next_value * (1 - o_batch)

            critic_loss = self.loss_fn(value, target_value) + 0.05 * self.loss_fn(value, next_value)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # 计算GAE
        buffer.compute_GAE(self.critic, self.gamma, self.lam)
        GAE = buffer.GAE_buffer.view(-1, 1)

        # Actor更新
        for _ in range(self.actor_update_frequency):
            idx = np.random.choice(len(state), self.batch_size, replace=False)
            s_batch, a_batch, ns_batch = state[idx], action[idx], next_state[idx]
            gae_batch, old_prob_batch = GAE[idx], old_prob[idx]

            # 对调并取反
            symmetry_s_batch = s_batch.clone()
            for i in range(0, 16, 2):
                symmetry_s_batch[:, i] = -s_batch[:, i + 1]
                symmetry_s_batch[:, i + 1] = -s_batch[:, i]

            # 计算新策略
            mu, std = self.actor(s_batch)
            symmetry_mu, _ = self.actor(symmetry_s_batch)
            new_prob = torch.distributions.Normal(mu, std).log_prob(a_batch).sum(dim=1, keepdim=True).exp()

            # PPO损失
            ratio = new_prob / (old_prob_batch + 1e-8)
            surr1 = ratio * gae_batch
            surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * gae_batch

            # 熵奖励和目标一致性损失
            entropy = torch.distributions.Normal(mu, std).entropy().mean()
            with torch.no_grad():
                mu_target, _ = self.actor(ns_batch)
                symmetry_mu_target = torch.empty_like(mu)
                for i in range(0, 8, 2):
                    symmetry_mu_target[:, i] = -mu[:, i + 1].detach()
                    symmetry_mu_target[:, i + 1] = -mu[:, i].detach()
                # 对调并取反结束
            actor_loss = -torch.min(surr1, surr2).mean() \
                         + 0.05 * self.loss_fn(mu, mu_target) \
                         - self.entropy_coef * entropy + \
                         0 * self.loss_fn(symmetry_mu, symmetry_mu_target)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        print(f"Experience Collected: {len(state)}, Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")

    def save_model(self):
        torch.save(self.actor.state_dict(), 'model/actor.pth')
        torch.save(self.critic.state_dict(), 'model/critic.pth')

    def save_each_epi_model(self):
        torch.save(self.actor.state_dict(), 'model/actor_f.pth')
        torch.save(self.critic.state_dict(), 'model/critic_f.pth')

    def load_model(self):
        self.actor.load_state_dict(torch.load('model/actor.pth'))
        self.critic.load_state_dict(torch.load('model/critic.pth'))

    def load_each_epi_model(self):
        self.actor.load_state_dict(torch.load('model/actor_f.pth'))
        self.critic.load_state_dict(torch.load('model/critic_f.pth'))
