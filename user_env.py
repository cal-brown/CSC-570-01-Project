"""
Environment for user testing. Allows you to use WASD to move Baba around.
"""

import os

env_name = "test-map-v1"
from environment import register_baba_env
import gym


if __name__ == "__main__":
    #level_path = os.path.join("baba-is-auto", "Resources", "Maps", "volcano.txt")
    level = 10
    env_name = "test-map-v%s" % str(level)
    env_path = os.path.join("levels","out","%s.txt" % str(level))
    env_template = register_baba_env(env_name, path=env_path, user_controls=True)
    env = gym.make(env_name)
    env.reset()
    while True:
        env.render()
    
