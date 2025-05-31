import random

from tqdm import tqdm
from agents import DynaQPlusAgent
from maze import MAZE_2, MAZE_3, Episode

seeds: list[int] = []
main_rng = random.Random(120)
while len(seeds) < 31:
    temp_seed = main_rng.randint(0, 2 ** 32 - 1)
    if temp_seed not in seeds:
        seeds.append(temp_seed)

rewards: dict[int, float] = {}
for s in tqdm(seeds):
    episode = Episode(maze=MAZE_2)  # Change to MAZE_3 for shortcut maze
    agent = DynaQPlusAgent(episode, n=50, rng=random.Random(s), kappa=1e-2)
    total = 0.0
    for _ in tqdm(range(5000)):  # Change to 10000 for shortcut maze
        _, _, r, _ = agent.step()
        total += r

    rewards[s] = total

reward_tuple_list = [(s, r) for s, r in rewards.items()]
reward_tuple_list.sort(key=lambda x: x[1])
print(reward_tuple_list)
print(reward_tuple_list[15])
