# BallPitNDF

This is the final project of Brian Lee and Eric Chen for [6.4210 (Robotic Manipulation)](https://manipulation.csail.mit.edu/Fall2024/schedule.html). In this project, we created a full-stack robotic system that uses Local Neural Descriptor Fields to grasp Baskets after only giving demos on Mugs and then do ball pouring. 

## Requirements and Development

Create a new conda environment using ```conda create -n ballpitndf python=3.11 pip```. Then activate this environment
with ```conda activate ballpitndf```. Install the required packages using ```pip install requirements.txt```. 
Alternatively use poetry.

To run the simulation run ```python -m main```.

## Documentation 

To get pretrained weights for evaluation, run ```sh scripts/get_lndf_sample.obj``` and ```sh scripts/get_weights.sh```.

A brief documentation of the major files in our code:

[*] `main.py`: The entrypoint to our project. We use argparse to allow a configurable number of balls to be dropped
into the mug. 
[*] `src/setup.py`: The `MakePandaManipulationStation` function creates a configurable (using Hydra) manipulation
station for the Franka Panda. This is then used in `main.py` to create a diagram for the pouring system.

### Perception

