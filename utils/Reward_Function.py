import torch
def abs_sum(target_value,value):
    if value.dim() > 1:
        return -torch.sum(torch.abs(target_value - value), dim=1)
    else:
        return -torch.abs(target_value - value)
def exp_sum(target_value,value):
    if value.dim() > 1:
        return torch.exp(torch.sum(torch.abs(target_value - value), dim=1))
    else:
        return  torch.exp(torch.abs(target_value - value))
def potential_reward(target_value,current_value,next_value,gamma=1):
    next_exp_reward = abs_sum(target_value,next_value)
    current_exp_reward = abs_sum(target_value,current_value)
    return next_exp_reward-current_exp_reward*gamma

def walking_phase(sine,cosine,L_foot_contact,R_foot_contact):
    # left_up = (sine>0) & (cosine>0)
    # left_down = (sine>0) & (cosine<0)
    # right_up = (sine<0) & (cosine<0)
    # right_down = (sine<0) &( cosine>0)
    #
    left = sine>0
    right = sine<0
    left = left.flatten()
    right = right.flatten()

    L_foot_contact = torch.any(L_foot_contact > 1e-5, dim=1)
    R_foot_contact = torch.any(R_foot_contact > 1e-5, dim=1)

    left_contact_but_right_swing = L_foot_contact & ~R_foot_contact

    # right_contact_but_left_swing = R_foot_contact & ~L_foot_contact
    # return ((left & right_contact_but_left_swing) + (right & left_contact_but_right_swing)).float()
    #
    reward = 2*(L_foot_contact != R_foot_contact).float()-1
    return reward


def foot_pos_generate(clock_signal):
    clock_signal_copy = torch.fmod(clock_signal.clone(),2*torch.pi)
    time =  torch.fmod(clock_signal.clone()/2/torch.pi,0.5)
    left_swing = (clock_signal_copy < torch.pi)

    poly_coef = [1.38313561e+01, -3.39458204e+01, 2.33525663e+01, -6.17086213e+00, 6.25020773e-01, 2.49770716e-04]
    des_z_pos = (poly_coef[0]*time**5+ \
                      poly_coef[1] * time ** 4 +\
    poly_coef[2] * time ** 3 +\
    poly_coef[3] * time ** 2 +\
    poly_coef[4] * time ** 1 +\
    poly_coef[5])*5

    des_z_pos = 0.15*torch.sin(2*torch.pi*time)
    des_z_vel = 0.15*2*torch.pi*torch.cos(2*torch.pi*time)

    L_swing_z = left_swing*des_z_pos
    R_swing_z = (~left_swing)*des_z_pos

    L_swing_z_dot = left_swing*des_z_vel
    R_swing_z_dot = (~left_swing)*des_z_vel

    return L_swing_z+0.07,R_swing_z+0.07,L_swing_z_dot,R_swing_z_dot

#
# t = torch.linspace(0,200,200).view(-1,1)/100
# print(t)
# clock = 2*torch.pi*t
# print(clock)
# a = foot_pos_generate(clock)[1].cpu().numpy().flatten()
# b = foot_pos_generate(clock)[0].cpu().numpy().flatten()
# c = foot_pos_generate(clock)[2].cpu().numpy().flatten()
# d = foot_pos_generate(clock)[3].cpu().numpy().flatten()
# print(a)
# from matplotlib.pyplot import *
# plot(a)
# plot(b)
# plot(c)
# plot(d)
# show()