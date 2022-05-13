import random
import sys
from time import sleep
sys.path.append("../baba-is-auto/Extensions/BabaRL/baba-babaisyou-v0")
import environment

import rendering

import pyBaba
import numpy as np
import gym
import random





class RandomAgent:
    
    def step(self, env: gym.Env) -> bool:
        """
        Makes a move in the environment
        Args:
            env: The environment where the agent will take an action
        Returns:
            Whether the environment is at a final state
        """
        action = random.choice(env.action_space)
        _, _, done, _ = env.step(action)
        return done
        


if __name__ == '__main__':
    env = gym.make('baba-babaisyou-v0')
    state = env.reset().reshape(1, -1, 9, 11)
    moves = 40
    done = False
    agent = RandomAgent()
    for i in range(moves):
        if done:
            break
        agent.step(env)
        env.render()
        sleep(0.2)