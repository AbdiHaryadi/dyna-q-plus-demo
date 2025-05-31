# Dyna-Q+ Demonstration

This program demonstrates Dyna-Q+ agent behavior in Dyna Maze environments, stated by Sutton and Barto (2018) in their book: _Reinforcement Learning: An Introduction_ (second edition). You can find it on page 164-168.

## Video
(TBA)

## Setup
1. Install Python.
2. (Optional) Create virtual environment.
3. Run `pip install -r requirements.txt`.

## Usage
Run `py visualization.py`.

## Advanced Configuration
You need to modify the Python code for further configuration. If you open `visualization.py`, you can change the value of `n` (number of planning steps) and `kappa` (Dyna-Q+ exploration factor). You can also change `ALPHA` (learning factor), `GAMMA` (discount rate), and `EPSILON` (for ɛ-greedy action selection) in `agents.py`. You can create your own maze—see some examples in `maze.py`. I also created Dyna-Q agent (without plus), but it's somehow not interesting to demonstrate because of "random walk" behavior at the beginning. For seed, I chose median case from 31 runs as a default for each case. You can generate another seed in `finding_good_seed.py`.

## Closing
I implemented Dyna-Q+ algorithm purely based on Sutton and Barto's book (2018). If there's any error on the implementation, feel free to open an issue. ;)
