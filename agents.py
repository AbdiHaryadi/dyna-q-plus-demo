from itertools import product
import random
import math
from maze import Action, Episode
from typing import Optional

ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.1

class Model:
    def __init__(self, all_positions: list[tuple[int, int]], all_actions: list[Action]):
        self.data = {sa: (0.0, (0, 0)) for sa in product(all_positions, all_actions)}
        self.observed: dict[tuple[int, int], set[Action]] = {}

    def update(self, pos: tuple[int, int], action: Action, reward: float, next_pos: tuple[int, int]):
        self.data[(pos, action)] = (reward, next_pos)
        if pos not in self.observed:
            self.observed[pos] = set()
        self.observed[pos].add(action)

    def sample(self, rng: Optional[random.Random] = None):
        if rng is None:
            pos = random.choice(list(self.observed.keys()))
            action = random.choice(list(self.observed[pos]))
        else:
            pos = rng.choice(list(self.observed.keys()))
            action = rng.choice(list(self.observed[pos]))
        reward, next_pos = self.data[(pos, action)]
        return pos, action, reward, next_pos
    
class ModelForDynaQPlus(Model):
    def __init__(self, all_positions: list[tuple[int, int]], all_actions: list[Action]):
        super().__init__(all_positions, all_actions)
        self.all_actions = all_actions
    
    def sample(self, rng: Optional[random.Random] = None):
        if rng is None:
            pos = random.choice(list(self.observed.keys()))
            action = random.choice(self.all_actions)
        else:
            pos = rng.choice(list(self.observed.keys()))
            action = rng.choice(self.all_actions)
        
        if action in self.observed.get(pos, set()):
            reward, next_pos = self.data[(pos, action)]
        else:
            reward = 0.0
            next_pos = pos
        
        return pos, action, reward, next_pos
    
class QTable:
    def __init__(self, all_positions: list[tuple[int, int]], all_actions: list[Action]):
        self.all_positions = all_positions
        self.all_actions = all_actions
        self.data = {sa: 0.0 for sa in product(all_positions, all_actions)}

    def get_value(self, pos: tuple[int, int], action: Action):
        return self.data[(pos, action)]
    
    def update(self, pos: tuple[int, int], action: Action, reward: float, next_pos: tuple[int, int]):
        self.data[(pos, action)] += ALPHA * (reward + GAMMA * max(self.data[(next_pos, a)] for a in self.all_actions) - self.data[(pos, action)])

    def get_all_actions(self):
        return self.all_actions
    
    def get_max_actions(self, pos: tuple[int, int]):
        action_score = max([self.data[(pos, a)] for a in self.all_actions])
        return set(a for a in self.all_actions if self.data[(pos, a)] == action_score)

def epsilon_greedy_action(q: QTable, pos: tuple[int, int], epsilon: float, rng: Optional[random.Random] = None):
    all_actions = q.get_all_actions()
    if rng is None:
        if random.random() < epsilon:
            action = random.choice(list(all_actions))
        else:
            action = max(all_actions, key=lambda a: (q.get_value(pos, a), random.random()))
    else:
        if rng.random() < epsilon:
            action = rng.choice(list(all_actions))
        else:
            action = max(all_actions, key=lambda a: (q.get_value(pos, a), rng.random()))

    return action

class DynaQAgent:
    def __init__(self, episode: Episode, n: int = 5, rng: Optional[random.Random] = None):
        self.episode = episode
        self.n = n

        all_positions = episode.get_all_positions()
        all_actions = episode.get_all_actions()
        self.q = QTable(all_positions, all_actions)
        self.model = Model(all_positions, all_actions)

        self.all_actions = all_actions
        self.rng = rng
    
    def step(self):
        # Step a
        pos = self.episode.get_current_position()

        # Step b
        action = epsilon_greedy_action(self.q, pos, epsilon=EPSILON, rng=self.rng)

        # Step c
        reward = self.episode.step(action)
        next_pos = self.episode.get_current_position()

        # Step d
        self.q.update(pos, action, reward, next_pos)

        # Step e
        self.model.update(pos, action, reward, next_pos)

        # Step f
        for _ in range(self.n):
            simulated_pos, simulated_action, simulated_r, simulated_next_pos = self.model.sample(rng=self.rng)
            self.q.update(simulated_pos, simulated_action, simulated_r, simulated_next_pos)

        return pos, action, reward, next_pos
    
    def get_max_actions(self, pos: tuple[int, int]):
        return self.q.get_max_actions(pos)
    
class DynaQPlusAgent(DynaQAgent):
    def __init__(self, episode: Episode, n: int = 5, rng: Optional[random.Random] = None, kappa=1e-3):
        super().__init__(episode, n, rng)
        self.kappa = kappa

        all_positions = episode.get_all_positions()
        all_actions = episode.get_all_actions()
        self.model = ModelForDynaQPlus(all_positions, all_actions)
        self.last_chosen = {sa: episode.get_current_timestep() - 1 for sa in product(all_positions, all_actions)}
    
    def step(self):
        # Step a
        pos = self.episode.get_current_position()

        # Step b
        action = epsilon_greedy_action(self.q, pos, epsilon=EPSILON, rng=self.rng)
        self.last_chosen[(pos, action)] = self.episode.get_current_timestep()

        # Step c
        reward = self.episode.step(action)
        next_pos = self.episode.get_current_position()

        # Step d
        self.q.update(pos, action, reward, next_pos)

        # Step e
        self.model.update(pos, action, reward, next_pos)

        # Step f
        for _ in range(self.n):
            simulated_pos, simulated_action, simulated_r, simulated_next_pos = self.model.sample(rng=self.rng)
            simulated_r += self.kappa * math.sqrt(self.episode.get_current_timestep() - self.last_chosen[(simulated_pos, simulated_action)])
            self.q.update(simulated_pos, simulated_action, simulated_r, simulated_next_pos)

        return pos, action, reward, next_pos
