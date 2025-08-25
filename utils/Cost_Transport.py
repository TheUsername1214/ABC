
import numpy as np
class Cost_Transport():
    def __init__(self):
        self.motor_energy = 0
        self.body_energy = 0

    def compute_motor_energy(self,torque,angular_vel,dt):
        self.motor_energy += np.sum(np.clip(torque*angular_vel*dt,0,111))
    def compute_body_energy(self,body_xy_pos,body_mass):
        self.body_energy = body_xy_pos*body_mass

    def reset(self):
        self.motor_energy = 0
        self.body_energy = 0

    def get_cot(self):
        return self.motor_energy / self.body_energy