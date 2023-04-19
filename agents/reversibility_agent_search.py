import os
import random
from time import sleep

from environment import register_baba_env

import pyBaba
import numpy as np
import gym
import random
from typing import List
import time
from enum import Enum

DEBUG = False

baba_dir_dict = {
    "U": pyBaba.Direction.UP,
    "R": pyBaba.Direction.RIGHT,
    "L": pyBaba.Direction.LEFT,
    "D": pyBaba.Direction.DOWN
}

class ReversibilityAgent:
    def __init__(self):
        self.step_planned = False
        self.moves = []

    def get_env_game_state(self, env):
        game_play_state = env.game.GetPlayState()
        game_objects = env.game.GetMap().GetObjects()
        game_rules = env.game.GetRuleManager().GetPropertyRules()
        game_player_icon = env.game.GetPlayerIcon()

        return (game_play_state, game_objects, game_rules, game_player_icon)

    def simulate(self, env: gym.Env, max_depth = 3):
        search_space = {0: [None]}
        depth = 0
        while depth < max_depth:
            roots = search_space[depth]
            if DEBUG:
                print(f"depth: {depth}, roots: {roots}")
            next_level = []
            for root in roots:
                if DEBUG:
                    print(f"root: {root}")
                if (root != None):
                    copied_game = root.prev_env.copy()
                    possible_env = gym.make(env_name)
                    possible_env.reset()
                    possible_env.set_game(copied_game)
                    for move in root.action_node.path:
                        possible_env.step(move)
                else:
                    possible_env = env

                if possible_env.game.GetPlayState() != pyBaba.PlayState.LOST:
                    children, winning_move = self.simulate_stage(root, possible_env)
                    if (winning_move != None):
                        print("FOUND WINNING MOVE")
                        self.moves = self.find_path(winning_move)
                        self.step_planned = True
                        return
                    next_level += children
                else:
                    root.lost = True
            depth += 1
            search_space[depth] = next_level
        possible_moves = search_space[1]
        non_losing_moves = list(filter(lambda n: not n.lost, possible_moves))
        target = random.choice(non_losing_moves)
        self.step_planned = True
        self.moves = target.action_node.path
        return

    def find_path(self, winning_move):
        path = []
        pointer = winning_move
        while pointer != None:
            path = pointer.action_node.path + path
            pointer = pointer.parent
        return path

    def simulate_stage(self, root, env: gym.Env):
        """
        Makes a move in the environment
        Args:
            env: The environment where the agent will take an action
        """
        possible_actions = self.find_actions(env)
        irreversible_actions = filter(lambda action: action.type == NodeType.IRREVERSIBLE, possible_actions)
        irreversible_nodes = [IrreversibleNode(node, root, env) for node in irreversible_actions]
        winning_move = self.find_winning_move(irreversible_nodes, env)
        return irreversible_nodes, winning_move


    def find_actions(self, env) -> List:
        player_positions = [tuple(p.flatten()) for p in self.get_your_positions(env)]
        root = ActionNode(player_positions)
        unexplored = [root]
        explored = {}
        while len(unexplored) > 0:
            node = unexplored.pop(0)
            if node.dist != 0:
                this_key = (tuple(node.pos), node.dir.name)
            else:
                this_key = (tuple(node.pos), None)
            if this_key in explored.keys():
                prev = explored[this_key]
                if node.dist < prev.dist: # take lesser of two distances if node already exists
                    explored[this_key] = node
            else:
                explored[this_key] = node
                if (node.type != NodeType.IRREVERSIBLE):
                  nbr_nodes = self.explore(node, env)
                  for nbr in nbr_nodes:
                      unexplored.append(nbr)
        return explored.values()

    def explore(self, node, env) -> List:
        nbrs = []
        this_pos = node.pos
        rule_manager = env.game.GetRuleManager()
        game_map = env.game.GetMap()
        map_height = game_map.GetHeight()
        map_width = game_map.GetWidth()
        convert = pyBaba.ConvertIconToText

        for dir in baba_dir_dict.keys():
          is_reversible = True
          is_stop = True
          new_pos = []
          for p in this_pos:
              new_p = move_dir(p, dir)
              if not within_bounds(new_p, map_height, map_width):
                  new_p = correct_pos(new_p, map_height, map_width)
                  is_reversible = False
              else:
                types = game_map.At(new_p[0], new_p[1]).GetTypes()
                if len(types) == 0:
                    is_stop = False
                any_stop = False
                for t in types:
                  text_type = convert(t)
                  if pyBaba.IsTextType(t):
                      is_reversible = False
                      is_stop = False
                  else:
                    rules = rule_manager.GetRules(text_type)
                    for rule in rules:
                        objects = rule.GetObjects()
                        left = objects[0]
                        right = objects[2]
                        #if the object text is on the left of the rule and anything but `YOU` is on the right - not reversible
                        if left.HasType(text_type) and not right.HasType(pyBaba.ObjectType.YOU):
                            is_reversible = False
                        if left.HasType(text_type) and right.HasType(pyBaba.ObjectType.STOP):
                            any_stop = True
                if not any_stop:
                    is_stop = False
              new_pos.append(new_p)
          if not is_stop:
            nbrs.append(ActionNode(new_pos, node.dist + 1, NodeType.REVERSIBLE if is_reversible else NodeType.IRREVERSIBLE, node.path, baba_dir_dict[dir]))
        return nbrs

    def find_winning_move(self, irreversible, env):
      goal_positions = self.get_goal_positions(env)
      for node in irreversible:
          positions = node.action_node.pos
          for p in positions:
              if p in goal_positions:
                  return node
      return None

    def step(self, env: gym.Env):
        action = self.moves.pop(0)
        if (len(self.moves) == 0):
            self.step_planned = False
        _, _, done, _ = env.step(action)
        return done

    def get_your_positions(self, env: gym.Env) -> List[np.array]:
        positions = env.game.GetMap().GetPositions(env.game.GetPlayerIcon())
        positions = [np.array(p) for p in positions]
        return positions

    def get_goal_positions(self, env: gym.Env) -> np.array:
        rule_manager = env.game.GetRuleManager()
        win_rules = rule_manager.GetRules(pyBaba.ObjectType.WIN)

        convert = pyBaba.ConvertTextToIcon
        win_positions = []
        game_map = env.game.GetMap()

        map_height = game_map.GetHeight()
        map_width = game_map.GetWidth()
        for y in range(map_height):
            for x in range(map_width):
                for win_rule in win_rules:

                    win_rule_type = win_rule.GetObjects()[0].GetTypes()[0]

                    if game_map.At(x, y).HasType(convert(win_rule_type)):
                        win_positions.append((x, y))

        return win_positions

class NodeType(Enum):
        START = 1
        REVERSIBLE = 2
        IRREVERSIBLE = 3

class ActionNode:
    def __init__(self, pos, dist=0, type=NodeType.START, path = [], dir = ''):
        self.pos = pos
        self.dist = dist
        self.type = type
        self.dir = dir
        if dir == '':
            self.path = path
        else:
            self.path = path + [dir]

    def __str__(self):
        return f"positions: {self.pos}, dir: {self.dir.name if isinstance(self.dir, pyBaba.Direction) else 'None'}"

    def __repr__(self):
        return str(self)

class IrreversibleNode:
    def __init__(self, action_node, parent, prev_env):
        self.action_node = action_node
        self.parent = parent
        self.prev_env = prev_env
        self.lost = False

    def __repr__(self):
        return f"Node: {self.action_node}, Parent: {self.parent.action_node if isinstance(self.parent, IrreversibleNode) else 'NA'} \n"


def move_dir(pos, dir):
   if (dir == 'U'):
       return (pos[0], pos[1]-1)
   elif (dir == 'D'):
       return (pos[0], pos[1]+1)
   elif (dir == 'L'):
       return (pos[0]-1, pos[1])
   elif (dir == 'R'):
       return (pos[0]+1, pos[1])
   else:
       raise Exception("bad direction passed")

def within_bounds(pos, height, width):
    return (pos[0] >= 0 and pos[0] < width and pos[1] >= 0 and pos[1] < height)

def correct_pos(pos, height, width):
    if pos[0] < 0:
        x = 0
    elif pos[0] >= width:
        x = width - 1
    else:
        x = pos[0]
    if pos[1] < 0:
        y = 0
    elif pos[1] >= height:
        y = height - 1
    else:
        y = pos[1]
    return (x,y)



if __name__ == "__main__":
    level = 10
    env_name = "test-map-v%s" % str(level)
    env_path = os.path.join("levels","out","%s.txt" % str(level))
    #random.seed(1)

    env_template = register_baba_env(env_name, path=env_path)
    env = gym.make(env_name)
    env.reset()
    # state = env.reset().reshape(1, -1, 9, 11)
    moves = 40
    done = False
    agent = ReversibilityAgent()

    num_sims = 0

    env.render()
    while not done:
        if (not agent.step_planned):
            num_sims += 1
            start_time = time.time()
            agent.simulate(env)
            print(f"Total time for simulation {num_sims}: {time.time() - start_time}s")
        done = agent.step(env)
        env.render()
        sleep(.2)
