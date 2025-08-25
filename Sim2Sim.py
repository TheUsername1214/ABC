import mujoco.viewer
from Config import *
from utils.NN import *
from utils.Useful_Function import *
from utils.Mujoco_Utils import *
from utils.Cost_Transport import *

COT = Cost_Transport()
state_dim = RobotConfig.CriticParam.state_dim
act_layers_num = RobotConfig.ActorParam.act_layers_num
actuator_num = RobotConfig.ActorParam.actuator_num
substep = RobotConfig.EnvParam.sub_step
dt = RobotConfig.EnvParam.dt
Kp = RobotConfig.RobotParam.Kp
Kd = RobotConfig.RobotParam.Kd
mujoco_dt = 0.001

device = torch.device("cuda:0")
actor = Actor(state_dim, act_layers_num, actuator_num).to(device)
actor.load_state_dict(torch.load('model/actor.pth'))

model = mujoco.MjModel.from_xml_path("SF_TRON1A/urdf/robot.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

v_cmd = FT([-0.5, 0, 0])

sim_time = FT([0])
swing_torque_list = []
swing_sim_time_list = []
stance_torque_list = []
stance_sim_time_list = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    i = 0
    while True:
        sim_time += mujoco_dt
        joint_angles = FT(get_actuator_joint_angles(model, data))  # 所有关节位置
        # 读取关节速度 (joint velocities)
        joint_velocities = FT(get_actuator_joint_velocities(model, data))  # 所有关节速度

        # 读取传感器数据 (包括IMU数据)
        sensor_data = data.qpos[3:7]
        euler_angle = get_euler_angle(FT([sensor_data])).flatten()
        if torch.any(torch.abs(euler_angle) > np.pi / 3) or i > 10000:
            i = 0
            body_pos = np.sum(np.sqrt((data.qpos[:2]) ** 2))

            COT.compute_body_energy(body_pos, 20.81)
            mujoco.mj_resetData(model, data)
            cot = COT.get_cot()
            break
        if (i % int(dt / mujoco_dt)) == 0:
            sine_clock = torch.sin(2 * torch.pi * sim_time)
            cosine_clock = torch.cos(2 * torch.pi * sim_time)
            state = torch.concatenate((joint_angles,
                                       joint_velocities,
                                       euler_angle,
                                       v_cmd,
                                       sine_clock,
                                       cosine_clock
                                       )).reshape(1, -1)


            angle, _ = actor(state)
            viewer.sync()
        i += 1
        print(sim_time)
        torque = Kp * (angle[0] - joint_angles) - Kd * joint_velocities

        torque[:6] = torque[:6].clip(-80, 80)
        torque[-2:] = torque[-2:].clip(-20, 20)
        data.ctrl[:] = torque.detach().cpu().numpy()

        mujoco.mj_step(model, data)

        stance_sim_time_list.append(sim_time.item())
        stance_torque_list.append(torque.detach().cpu().numpy()[-1])

        joint_velocities = get_actuator_joint_velocities(model, data)  # 所有关节速度
        COT.compute_motor_energy(torque.detach().cpu().numpy()[-1], joint_velocities, 0.001)

print(cot)
from matplotlib.pyplot import *

plot(np.array(stance_sim_time_list), np.array(stance_torque_list), "g")
show()
