import os
# from .render import PyBulletRecorder

from gym.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="PandaReach{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPush{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaSlide{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPickAndPlace{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )
        
        register(
            id="PandaStack{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaStackEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="PandaFlip{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaFlipEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )
for i in range(1,7):
    register(
        id="PandaTower-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerEnv",
        kwargs={"reward_type": 'sparse', "control_type": 'ee', 'num_blocks': i},
        max_episode_steps=50*i,
    )
    register(
        id="PandaTowerBimanual-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'hand_num_mix'}
    )
    register(
        id="PandaTowerBimanualParallel-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'swarm', "parallel_robot": True}
    )
    register(
        id="PandaTowerBimanualParallelExchange-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'swarm', "parallel_robot": True, 'os_rate': 1, 'exchange_only': True}, 
        max_episode_steps = 160
    )
    register(
        id="PandaTowerBimanualSlow-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'num_blocks', 'max_move_per_step': 0.02}
    )
    register(
        id="PandaTowerBimanualSlowNoise-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'num_blocks', 'max_move_per_step': 0.02, 'noise_obs': True}
    )
    for j in range(1, 7):
        register(
            id=f"PandaTowerBimanualMaxHandover{j}-v{i}",
            entry_point="panda_gym.envs:PandaTowerBimanualEnv",
            kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'hand_num_mix', 'max_num_need_handover':j}
        )
        register(
            id=f"PandaTowerBimanualMaxHandover{j}Slow-v{i}",
            entry_point="panda_gym.envs:PandaTowerBimanualEnv",
            kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'num_blocks', 'max_num_need_handover':j, 'max_move_per_step': 0.02}
        )
        register(
            id=f"PandaTowerBimanualMaxHandover{j}SlowNoise-v{i}",
            entry_point="panda_gym.envs:PandaTowerBimanualEnv",
            kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'num_blocks', 'max_num_need_handover':j, 'max_move_per_step': 0.02, 'noise_obs': True}
        )
    register(
        id="PandaTowerBimanualReachOnce-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'num_blocks', 'reach_once': True}
    )
    register(
        id="PandaTowerBimanualSingleSide-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'num_blocks', 'single_side': True}
    )
    register(
        id="PandaTowerBimanualReachOnceMix-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'mix', 'reach_once': True}
    )
    register(
        id="PandaTowerBimanualGravity-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'gravity'},
    )
    register(
        id="PandaTowerBimanualOtherSide-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'other_side'}
    )
    register(
        id="PandaRearrange-v"+str(i),
        entry_point="panda_gym.envs:PandaRearrangeEnv",
        kwargs={"num_blocks": i, "unstable_mode": False}
    )
    register(
        id="PandaRearrangeUnstable-v"+str(i),
        entry_point="panda_gym.envs:PandaRearrangeEnv",
        kwargs={"num_blocks": i, "unstable_mode": True}
    )
register(
    id="PandaRearrange-v0",
    entry_point="panda_gym.envs:PandaRearrangeEnv",
)
register(
    id="PandaPickAndPlace2-v2",
    entry_point="panda_gym.envs:PandaPickAndPlace2Env",
    kwargs={'curriculum_type': 'num_obj'},
)
register(
    id="PandaLiftBimanual-v0",
    entry_point="panda_gym.envs:PandaLiftBimanualEnv",
    max_episode_steps=50,
)
register(
    id="PandaReachBimanual-v0",
    entry_point="panda_gym.envs:PandaReachBimanualEnv",
    max_episode_steps=50,
)
register(
    id="PandaRelativePNPBimanual-v0",
    entry_point="panda_gym.envs:PandaReachBimanualEnv",
    kwargs={'has_object': True, 'obj_not_in_hand_rate': 1},
    max_episode_steps=50,
)
register(
    id="PandaRelativePNPBimanualObjInHand-v0",
    entry_point="panda_gym.envs:PandaReachBimanualEnv",
    kwargs={'has_object': True, 'obj_not_in_hand_rate': 0},
    max_episode_steps=50,
)
register(
    id="PandaPNPBimanual-v0",
    entry_point="panda_gym.envs:PandaReachBimanualEnv",
    kwargs={'has_object': True, 'obj_not_in_hand_rate': 1, 'absolute_pos': True, 'blender_record': True},
    max_episode_steps=50,
)
register(
    id="PandaPNPBimanualObjInHand-v0",
    entry_point="panda_gym.envs:PandaReachBimanualEnv",
    kwargs={'has_object': True, 'obj_not_in_hand_rate': 0, 'absolute_pos': True},
    max_episode_steps=50,
)
register(
    id="PandaTowerBimanualShort-v2",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 2},
    max_episode_steps=50,
)
register(
    id="PandaTowerBimanualBound-v2",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 2, 'use_bound': True},
    max_episode_steps=100,
)
register(
    id="PandaTowerBimanualMusk-v2",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 2, 'use_musk': True, 'curriculum_type': 'musk'},
    max_episode_steps=140,
)
register(
    id="PandaTowerBimanualInHand-v2",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 2, 'curriculum_type': 'in_hand'},
    max_episode_steps=100,
)
register(
    id="PandaTowerBimanualInHand-v1",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 1, 'curriculum_type': 'in_hand'},
    max_episode_steps=50,
)
register(
    id="PandaTowerBimanualGoalInObj-v2",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 2, 'curriculum_type': 'goal_in_obj'},
    max_episode_steps=140,
)
register(
    id="PandaTowerBimanualNumBlocks-v2",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 1, 'curriculum_type': 'num_blocks'}
)
for i in range(7):
    register(
        id="PandaTowerBimanualSharedOpSpace-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks':1 if i==0 else i, 'shared_op_space': True, 'curriculum_type': 'num_blocks'},
    )
    register(
        id="PandaTowerBimanualNoGap-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks':1 if i==0 else i, 'shared_op_space': True, 'curriculum_type': 'num_blocks', 'gap_distance': 0},
    )
    register(
        id="PandaTowerBimanualNoGapMix-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks':1 if i==0 else i, 'shared_op_space': True, 'curriculum_type': 'os_num_mix', 'gap_distance': 0},
    )
    register(
        id="PandaTowerBimanualDelay-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks':1 if i==0 else i, 'curriculum_type': 'other_side', 'max_delay_steps': 30}
    )
register(
    id="PandaAssembleBimanual-v2",
    entry_point="panda_gym.envs:PandaAssembleBimanualEnv",
    kwargs={'obj_not_in_hand_rate': 0.5, 'obj_not_in_plate_rate':0.5}
)
register(
    id="PandaTowerBimanualOsNumMix-v1",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 1, 'curriculum_type': 'os_num_mix'}
)
register(
    id="PandaTowerBimanualHandNumMix-v1",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 1, 'curriculum_type': 'hand_num_mix'}
)