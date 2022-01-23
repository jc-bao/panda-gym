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
        kwargs={"control_type": 'ee', 'num_blocks': i},
        max_episode_steps=50*i,
    )
    register(
        id="PandaTowerBimanualGravity-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'gravity'},
        max_episode_steps=50*i,
    )
    register(
        id="PandaTowerBimanualOtherSide-v"+str(i),
        entry_point="panda_gym.envs:PandaTowerBimanualEnv",
        kwargs={"control_type": 'ee', 'num_blocks': i, 'curriculum_type': 'other_side'},
        max_episode_steps=50*i,
    )

register(
    id="PandaRearrange-v2",
    entry_point="panda_gym.envs:PandaRearrangeEnv",
    max_episode_steps=50,
)
register(
    id="PandaRearrange-v0",
    entry_point="panda_gym.envs:PandaRearrangeEnv",
    max_episode_steps=50,
)
register(
    id="PandaPickAndPlace2-v2",
    entry_point="panda_gym.envs:PandaPickAndPlace2Env",
    kwargs={'curriculum_type': 'num_obj'},
    max_episode_steps=50,
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
    max_episode_steps=50,
)
register(
    id="PandaTowerBimanualInHand-v2",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 2, 'curriculum_type': 'in_hand'},
    max_episode_steps=50,
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
    max_episode_steps=50,
)
register(
    id="PandaTowerBimanualGoalZ-v2",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 2, 'curriculum_type': 'goal_z'},
    max_episode_steps=50,
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
    id="PandaTowerBimanualNoGapMixCurri-v0",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks':1, 'shared_op_space': True, 'curriculum_type': 'mix_2', 'gap_distance': 0},
)
register(
    id="PandaTowerBimanualDelay-v0",
    entry_point="panda_gym.envs:PandaTowerBimanualEnv",
    kwargs={"control_type": 'ee', 'num_blocks': 1, 'curriculum_type': 'mix', 'max_delay_steps': 30}
)