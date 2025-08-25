def setup_world_physics(World,device,backend,dt,sub_step):
    """配置物理世界参数"""
    world = World(device=device, backend=backend)
    world.set_simulation_dt(physics_dt=dt / sub_step, rendering_dt=dt / sub_step)
    world.get_physics_context().set_gpu_max_rigid_patch_count(60010)
    world.get_physics_context().set_gpu_max_rigid_contact_count(111111)
    world.get_physics_context().set_gpu_found_lost_pairs_capacity(26161311)
    world.get_physics_context().set_gpu_found_lost_aggregate_pairs_capacity(99609250)
    world.get_physics_context().set_gpu_total_aggregate_pairs_capacity(8002000)
    world.get_physics_context().enable_gpu_dynamics(True)
    return world

