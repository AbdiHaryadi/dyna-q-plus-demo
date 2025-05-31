import tkinter
import random
from pygame import mixer
mixer.init()

from agents import DynaQAgent, DynaQPlusAgent
from maze import MAZE_3, Action, MAZE_2, MAZE_1, DynamicMaze

HEADER_SIZE = 32
CELL_SIZE = 64
BORDER_SIZE = 16

class PolicyDrawing:
    def __init__(self, canvas: tkinter.Canvas, x: int, y: int):
        self.canvas = canvas
        self.right = canvas.create_text(x + CELL_SIZE / 4, y, anchor=tkinter.CENTER, text="→", font=("Arial", 16))
        self.left = canvas.create_text(x - CELL_SIZE / 4, y, anchor=tkinter.CENTER, text="←", font=("Arial", 16))
        self.up = canvas.create_text(x, y - CELL_SIZE / 4, anchor=tkinter.CENTER, text="↑", font=("Arial", 16))
        self.down = canvas.create_text(x, y + CELL_SIZE / 4, anchor=tkinter.CENTER, text="↓", font=("Arial", 16))

        self.hide_all()

    def hide_all(self):
        canvas = self.canvas
        canvas.itemconfig(self.right, state="hidden")
        canvas.itemconfig(self.left, state="hidden")
        canvas.itemconfig(self.up, state="hidden")
        canvas.itemconfig(self.down, state="hidden")

    def update(self, actions: set[Action]):
        self.hide_all()
        if len(actions) == 4:
            return

        for a in actions:
            match a:
                case "left":
                    self.show(self.left)
                case "right":
                    self.show(self.right)
                case "up":
                    self.show(self.up)
                case "down":
                    self.show(self.down)
                case _:
                    raise ValueError(f"Unknown action: {a}")

    def show(self, item_id: int):
        self.canvas.itemconfig(item_id, state="normal")

class SimulationCanvas:
    def __init__(self, root: tkinter.Tk, maze: DynamicMaze, title: str = ""):
        row_count = maze.get_row_count()
        column_count = maze.get_column_count()
        canvas = tkinter.Canvas(
            root,
            bg="white",
            height=row_count * CELL_SIZE + BORDER_SIZE * 2 + HEADER_SIZE,
            width=column_count * CELL_SIZE + BORDER_SIZE * 2
        )
        canvas.pack()

        half_cell_size = CELL_SIZE / 2

        policy_drawing_map: dict[tuple[int, int], PolicyDrawing] = {}
        rectangle_map: dict[tuple[int, int], int] = {}

        for r in range(row_count):
            y0 = BORDER_SIZE + r * CELL_SIZE + HEADER_SIZE
            for c in range(column_count):
                x0 = BORDER_SIZE + c * CELL_SIZE

                rectangle = canvas.create_rectangle(x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE)
                cell = maze.get_cell(0, (r, c))
                if cell == "X":
                    canvas.itemconfig(rectangle, fill="gray")
                elif cell == "S":
                    canvas.itemconfig(rectangle, fill="orange")
                elif cell == "G":
                    canvas.itemconfig(rectangle, fill="cyan")
                rectangle_map[(r, c)] = rectangle

                pd = PolicyDrawing(canvas, x0 + half_cell_size, y0 + half_cell_size)
                policy_drawing_map[(r, c)] = pd

                if cell == "S":
                    canvas.create_text(x0 + CELL_SIZE / 2, y0 + CELL_SIZE / 2, anchor=tkinter.CENTER, text="S", font=("Arial", 16))
                elif cell == "G":
                    canvas.create_text(x0 + CELL_SIZE / 2, y0 + CELL_SIZE / 2, anchor=tkinter.CENTER, text="G", font=("Arial", 16))

        episode = maze.create_episode()

        r, c = episode.get_current_position()
        x0 = BORDER_SIZE + (c + 1/4) * CELL_SIZE
        y0 = BORDER_SIZE + (r + 1/4) * CELL_SIZE + HEADER_SIZE
        agent_rect = canvas.create_rectangle(x0, y0, x0 + half_cell_size, y0 + half_cell_size, fill="black")

        x0 = BORDER_SIZE
        y0 = (BORDER_SIZE + HEADER_SIZE) / 2
        step_text = canvas.create_text(x0, y0, anchor=tkinter.W, text=f"Timestep: {episode.get_current_timestep()}")
        
        x0 = BORDER_SIZE + column_count * CELL_SIZE / 2
        y0 = (BORDER_SIZE + HEADER_SIZE) / 2
        total_reward_text = canvas.create_text(x0, y0, anchor=tkinter.CENTER, text=f"Total reward: 0")
        
        x0 = BORDER_SIZE + column_count * CELL_SIZE
        y0 = (BORDER_SIZE + HEADER_SIZE) / 2
        canvas.create_text(x0, y0, anchor=tkinter.E, text=title)

        self.root = root
        self.canvas = canvas
        self.maze = maze
        self.episode = episode
        self.agent_rect = agent_rect
        self.policy_drawing_map = policy_drawing_map
        self.rectangle_map = rectangle_map
        self.step_text = step_text
        self.total_reward_text = total_reward_text
        self.move_sounds = {
            a: mixer.Sound(f"sounds/{a}.ogg")
            for a in ["left", "right", "up", "down"]
        }
        self.reward_sound = mixer.Sound("sounds/coin.wav")
        self.refresh_period = 50
        self.total_reward = 0

    def update(self, agent: DynaQAgent, max_timestep: int):
        prev_pos, action, reward, new_pos = agent.step()
        timestep = self.episode.get_current_timestep()
        if max_timestep == -1 or timestep < max_timestep:
            self.root.after(self.refresh_period, lambda: self.update(agent, max_timestep))

        self.total_reward += int(reward)

        (r, c) = prev_pos

        x0 = BORDER_SIZE + (c + 1/4) * CELL_SIZE
        y0 = BORDER_SIZE + (r + 1/4) * CELL_SIZE + HEADER_SIZE

        canvas = self.canvas
        canvas.moveto(self.agent_rect, x0, y0)


        for r in range(self.maze.get_row_count()):
            for c in range(self.maze.get_column_count()):
                pos = (r, c)
                match self.episode.get_cell(pos):
                    case ".":
                        canvas.itemconfig(self.rectangle_map[pos], fill="white")
                    case "X":
                        canvas.itemconfig(self.rectangle_map[pos], fill="gray")

                self.policy_drawing_map[pos].update(agent.get_max_actions(pos))
        
        canvas.itemconfig(self.step_text, text=f"Timestep: {timestep}")
        canvas.itemconfig(self.total_reward_text, text=f"Total reward: {self.total_reward}")

        if reward > 0:
            self.reward_sound.play()
        
        if prev_pos != new_pos:
            self.move_sounds[action].play()

    def run(self, agent: DynaQAgent, max_timestep: int = -1):
        self.root.after(self.refresh_period, lambda: self.update(agent, max_timestep))

    def get_episode(self):
        return self.episode

if __name__ == "__main__":
    print("Select maze case:")
    print("(1) Blocking Maze")
    print("(2) Shortcut Maze")

    maze = None
    max_timestep = -1
    title = ""
    seed = -1
    while maze is None:
        maze_no = input("Answer (1-2): ")
        match maze_no:
            case "1":
                maze = MAZE_2
                max_timestep = 3000
                title = "Blocking Maze (3000 steps)"
                seed = 1109577984
            case "2":
                maze = MAZE_3
                max_timestep = 6000
                title = "Shortcut Maze (6000 steps)"
                seed = 1360610827
            case _:
                print(f"(!) Invalid answer: {maze_no}")

    root = tkinter.Tk()

    # Default configuration
    n = 50
    kappa = 1e-3
    start_delay_ms = 1000

    canvas = SimulationCanvas(root, maze=maze, title=title)
    agent = DynaQPlusAgent(canvas.get_episode(), n=n, rng=random.Random(seed), kappa=kappa)
    root.after(start_delay_ms, lambda: canvas.run(agent, max_timestep=max_timestep))
    root.mainloop()
