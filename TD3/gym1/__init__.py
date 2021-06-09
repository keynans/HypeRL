from gym.envs.registration import register

register(
    'MyAnt-v0',
    entry_point='gym1.envs.ant:AntEnv',
    kwargs={},
    max_episode_steps=1000
)