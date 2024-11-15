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

* `main.py`: The entrypoint to our project. We use argparse to allow a configurable number of balls to be dropped
into the mug. 
* `src/setup.py`: The `MakePandaManipulationStation` function creates a configurable (using Hydra) manipulation
station for the Franka Panda. This is then used in `main.py` to create a diagram for the pouring system.

### Perception

* `merge_point_clouds.py`: Introuces the `MergePointClouds` system which combines the point clouds of a station 
into a single point cloud. For now it only concatenates and assumes calibrated point clouds (TODO: add camera
calibration for uncalibrated cameras using ICP)

### Debugging

All debug utilities are in `debug/`. The main files are:
* `visualize_utils.py` which introuduces `visualize_camera_images` and `visualize_depth_images` to see the camera 
results from a built station, `visualize_point_cloud` which plots a point cloud (given as an 3xN numpy array) in 
plotly, and `visualize_diagram` which plots the system diagram of a diagram (without need for iPython) along with the
ability to save the SVG file onto your computer.

