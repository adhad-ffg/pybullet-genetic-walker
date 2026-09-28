Here is the clean and professionally formatted version of your README in English, without any emojis.

## pybullet-genetic-walker
Simulation of multi-joint walking and obstacle traversal using a Genetic Algorithm (GA) and PyBullet.

## Installation
To run this simulation, you need to install pybullet and numpy. You can install them via pip:
```bash
pip install pybullet numpy
```

## File Structure
```txt
config.py | Contains global parameters (population size, mutation rate, simulation steps, etc.). 

network.py | Defines the forward-propagation neural network serving as the agent's brain (genes). 

environment.py | Manages the PyBullet interface, humanoid control, and fitness evaluation. 

ga.py | Implements the core Genetic Algorithm logic (selection, crossover, mutation). 

main.py | The main loop orchestrating generations and simulation phases. 
```

## Usage## Running the Simulation
Execute the main evolution loop by running the following command:

python main.py

## Tip for Faster Training

* To speed up the training process, set RENDER_SIMULATION = False in config.py to disable the GUI rendering.

## license
MITlicsnse
