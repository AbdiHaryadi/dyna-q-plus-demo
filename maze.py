from typing import Literal

Action = Literal["up", "down", "right", "left"]

class DynamicMaze:
    def get_row_count(self) -> int:
        raise NotImplementedError
    
    def get_column_count(self) -> int:
        raise NotImplementedError

    def get_cell(self, timestep: int, pos: tuple[int, int]) -> Literal[".", "X", "S", "G"]:
        raise NotImplementedError
    
    def all_pos(self):
        for r in range(self.get_row_count()):
            for c in range(self.get_column_count()):
                yield (r, c)

    def start_pos(self, timestep: int = 0):
        for pos in self.all_pos():
            if self.get_cell(timestep, pos) == "S":
                return pos
        
        raise ValueError("Cannot found S")
    
    def step(self, timestep: int, pos: tuple[int, int], action: Action):
        r, c = pos
        match action:
            case "up":
                if r > 0:
                    r -= 1
            case "down":
                max_r = self.get_row_count() - 1
                if r < max_r:
                    r += 1
            case "left":
                if c > 0:
                    c -= 1
            case "right":
                max_c = self.get_column_count() - 1
                if c < max_c:
                    c += 1
            case _:
                raise ValueError(f"Unknown action: {action}")
            
        cell = self.get_cell(timestep, (r, c))
        if cell == "X":
            return 0.0, pos  # Unchanged
        
        if cell == "G":
            return 1.0, self.start_pos(timestep)
        
        return 0.0, (r, c)
    
    def all_actions(self):
        yield "up"
        yield "right"
        yield "down"
        yield "left"

    def create_episode(self):
        return Episode(self)

class FixedMaze(DynamicMaze):
    def __init__(self, data: list[list[str]]):
        self.data = data
    
    def get_row_count(self) -> int:
        return len(self.data)
    
    def get_column_count(self) -> int:
        return len(self.data[0])

    def get_cell(self, timestep: int, pos: tuple[int, int]) -> Literal[".", "X", "S", "G"]:
        r, c = pos
        return self.data[r][c]
    
class ChangedMaze(DynamicMaze):
    def __init__(self, first_maze: DynamicMaze, second_maze: DynamicMaze, change_timestep: int):
        self.first_maze = first_maze
        self.second_maze = second_maze
        self.change_timestep = change_timestep
    
    def get_row_count(self) -> int:
        return self.first_maze.get_row_count()
    
    def get_column_count(self) -> int:
        return self.second_maze.get_column_count()

    def get_cell(self, timestep: int, pos: tuple[int, int]) -> Literal[".", "X", "S", "G"]:
        if timestep < self.change_timestep:
            return self.first_maze.get_cell(timestep, pos)
        else:
            return self.second_maze.get_cell(timestep, pos)

class Episode:
    def __init__(self, maze: DynamicMaze):
        self.maze = maze
        self.current_pos = maze.start_pos()
        self.timestep = 0

    def get_all_positions(self):
        return list(self.maze.all_pos())
    
    def get_all_actions(self):
        return list(self.maze.all_actions())

    def get_current_position(self):
        return self.current_pos

    def step(self, action: Action):
        r, next_pos = self.maze.step(self.timestep, self.current_pos, action)
        self.current_pos = next_pos
        self.timestep += 1
        return r
    
    def get_cell(self, pos: tuple[int, int]):
        return self.maze.get_cell(self.timestep, pos)
    
    def get_current_timestep(self):
        return self.timestep

MAZE_1 = FixedMaze("""
.......XG
..X....X.
S.X....X.
..X......
.....X...
.........
""".strip().split())

maze_2_a = FixedMaze("""
........G
.........
.........
XXXXXXXX.
.........
...S.....
""".strip().split())

maze_2_b = FixedMaze("""
........G
.........
.........
.XXXXXXXX
.........
...S.....
""".strip().split())

MAZE_2 = ChangedMaze(maze_2_a, maze_2_b, 1000)

maze_3_a = maze_2_b
maze_3_b = FixedMaze("""
........G
.........
.........
.XXXXXXX.
.........
...S.....
""".strip().split())
MAZE_3 = ChangedMaze(maze_3_a, maze_3_b, 3000)
