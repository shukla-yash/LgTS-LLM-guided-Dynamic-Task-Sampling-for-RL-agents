from gymnasium.envs.registration import register 

# Environment for debugging
register(
    id='FiremanGrid-ExtinguishFire-v0',
    entry_point='firemangrid.envs:ExtinguishFireEnv',
)

register(
    id='FiremanGrid-MoveDebris-v0',
    entry_point='firemangrid.envs:MoveDebrisEnv',
)

register(
    id='FiremanGrid-RescueSurvivor-v0',
    entry_point='firemangrid.envs:RescueSurvivorEnv',
) 

register(
    id='FiremanGrid-OpenDoor-v0',
    entry_point='firemangrid.envs:OpenDoorEnv',
) 

# Fireman Whole Tasks

# Starting from start
register(
    id='FiremanGrid-Start2Key-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'start2key'}
)

register(
    id='FiremanGrid-Start2Door-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'start2door'}
)

register(
    id='FiremanGrid-Start2FireExtinguisher-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'start2fireextinguisher'}
)

register(
    id='FiremanGrid-Start2Fire-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'start2fire'}
) 

register(
    id='FiremanGrid-Start2Debris-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'start2debris'}
) 

register(
    id='FiremanGrid-Start2Survivor-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'start2survivor'}
) 

# Starting from key
register(
    id='FiremanGrid-Key2Start-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'key2start'}
)

register(
    id='FiremanGrid-Key2Door-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'key2door'}
)

register(
    id='FiremanGrid-Key2FireExtinguisher-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'key2fireextinguisher'}
) 

register(
    id='FiremanGrid-Key2Fire-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'key2fire'}
)

register(
    id='FiremanGrid-Key2Debris-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'key2debris'}
) 

register(
    id='FiremanGrid-Key2Survivor-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'key2survivor'}
) 

# Starting from door 
register(
    id='FiremanGrid-Door2Start-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'door2start'}
)

register(
    id='FiremanGrid-Door2Key-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'door2key'}
)

register(
    id='FiremanGrid-Door2FireExtinguisher-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'door2fireextinguisher'}
) 

register(
    id='FiremanGrid-Door2Fire-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'door2fire'}
) 

register(
    id='FiremanGrid-Door2Debris-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'door2debris'}
) 

register(
    id='FiremanGrid-Door2Survivor-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'door2survivor'}
) 

# Starting from fire extinguisher 
register(
    id='FiremanGrid-FireExtinguisher2Start-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fireextinguisher2start'}
)

register(
    id='FiremanGrid-FireExtinguisher2Key-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fireextinguisher2key'}
)

register(
    id='FiremanGrid-FireExtinguisher2Door-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fireextinguisher2door'}
)

register(
    id='FiremanGrid-FireExtinguisher2Fire-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fireextinguisher2fire'}
)

register(
    id='FiremanGrid-FireExtinguisher2Debris-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fireextinguisher2debris'}
)

register(
    id='FiremanGrid-FireExtinguisher2Survivor-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fireextinguisher2survivor'}
)

# Starting from fire
register(
    id='FiremanGrid-Fire2Start-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fire2start'}
)

register(
    id='FiremanGrid-Fire2Key-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fire2key'}
) 

register(
    id='FiremanGrid-Fire2Door-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fire2door'}
) 

register(
    id='FiremanGrid-Fire2FireExtinguisher-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fire2fireextinguisher'}
)

register(
    id='FiremanGrid-Fire2Debris-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fire2debris'}
)

register(
    id='FiremanGrid-Fire2Survivor-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'fire2survivor'}
)

# Starting from debris
register(
    id='FiremanGrid-Debris2Start-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'debris2start'}
)

register(
    id='FiremanGrid-Debris2Key-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'debris2key'}
)

register(
    id='FiremanGrid-Debris2Door-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'debris2door'}
)

register(
    id='FiremanGrid-Debris2FireExtinguisher-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'debris2fireextinguisher'}
)

register(
    id='FiremanGrid-Debris2Fire-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'debris2fire'}
)

register(
    id='FiremanGrid-Debris2Survivor-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'debris2survivor'}
) 

# Starting from survivor
register(
    id='FiremanGrid-Survivor2Start-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'survivor2start'}
)

register(
    id='FiremanGrid-Survivor2Key-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'survivor2key'}
)

register(
    id='FiremanGrid-Survivor2Door-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'survivor2door'}
)

register(
    id='FiremanGrid-Survivor2FireExtinguisher-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'survivor2fireextinguisher'}
)

register(
    id='FiremanGrid-Survivor2Fire-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'survivor2fire'}
)

register(
    id='FiremanGrid-Survivor2Debris-v0',
    entry_point='firemangrid.envs:FiremanWholeEnv',
    kwargs={'task': 'survivor2debris'}
)