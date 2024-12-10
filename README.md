# BallPitNDF: Few-Shot Pouring with Local Neural Descriptor Fields

## [Report](https://www.dropbox.com/scl/fi/01kluje9vmr3jcukoggf4/ballpitndf.pdf?rlkey=qf6e1uaunnxw41y3309w394f7&st=j75hw6fa&dl=0)

This is the final project of Brian Lee and Eric Chen for [6.4210 (Robotic Manipulation)](https://manipulation.csail.mit.edu/Fall2024/schedule.html). In this project, we created a full-stack robotic system that uses Local Neural Descriptor Fields to grasp Baskets after only giving demos on Mugs and then do ball pouring.

## Requirements and Development

Create a new conda environment using `conda create -n ballpitndf python=3.11 pip`. Then activate this environment
with `conda activate ballpitndf`. Install the required packages using `pip install -r requirements.txt`.
Then, install torch_scatter with `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+${CUDA_VERSION}.html`
where you can replace `${CUDA_VERSION}` with your version of CUDA (supported: `cpu` and `cu121`). Python version *must* 
be 3.11 as either torch_scatter or drake doesn't support other versions.

### Mac/Linux Users:

An additional pykdtree library must be installed on Unix-based devices as:
`pip install --no-binary pykdtree --force pykdtree`.

### Windows/CUDA Support:
Finally, install `gcc` with `conda install -c conda-forge gcc=12.1.0` and get `Xvbf` with 
`sudo apt-get install xvfb` if you have a device with CUDA (along with any other additional nvidia cuda drivers
neccesary).

### Texture/Demo/Checkpoint Installation.

Please run `source scripts/setup.sh`.

## Running the Simulation

To run the simulation run `python -m main`.

## Running Experiments

To run the experiment run `python -m run_experiments +num_runs=${NUM_RUNS} +duration=${DURATION}` replacing `${NUM_RUNS}` with the 
number of runs you will run (we use 100 in our paper) and `${DURATION}` for the amount of time to be used (we typicall use 35 seconds).

## Documentation

To get pretrained weights for evaluation, run `sh scripts/get_lndf_sample.obj` and `sh scripts/get_weights.sh`.

A brief documentation of the major files in our code:

- `main.py`: The entrypoint to our project. We use argparse to allow a configurable number of balls to be dropped
  into the mug.
- `src/setup.py`: The `MakePandaManipulationStation` function creates a configurable (using Hydra) manipulation
  station for the Franka Panda. This is then used in `main.py` to create a diagram for the pouring system.

### Perception

* `merge_point_clouds.py`: Introuces the `MergePointClouds` system which combines the point clouds of a station 
into a single point cloud. The crop is done without cheat ports using a combination of the 
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) algorithms and the 
[Segment Anything](https://github.com/facebookresearch/segment-anything) algorithms for segmentation. 

### Grasping

* `grasper.py`: Contains the `LNDFGrasper` system that does an energy optimization to find suitable grasp poses 
to grasp objects (such as the basket). The main call is to `pose_selector` which is in the `lndf_robot` module.
* `pose_selector.py`: Contains the `LocalNDF` wrapper class that makes calls to models in the `lndf_robot` submodule
to do the energy minimization from the given demonstrations.

### Kinematics

* `diff_ik.py`: Contains a custom Differential IK class that sets joints limits for the panda arm (the hand is done
manually rather than through an InverseDynamicsDriver).
* `grasp_utils.py`: Utility functions to create grasp trajectories from a set of keypoints.
* `planner.py`: Planner System that plans the whole pouring process splitting into 8 distinct states. 

### Debugging

All debug utilities are in `debug/`. The main files are:
* `visualize_utils.py` which introuduces `visualize_camera_images` and `visualize_depth_images` to see the camera 
results from a built station, `visualize_point_cloud` which plots a point cloud (given as an 3xN numpy array) in 
plotly, and `visualize_diagram` which plots the system diagram of a diagram (without need for iPython) along with the
ability to save the SVG file onto your computer. 

- `visualize_utils.py` which introuduces `visualize_camera_images` and `visualize_depth_images` to see the camera
  results from a built station, `visualize_point_cloud` which plots a point cloud (given as an 3xN numpy array) in
  plotly, and `visualize_diagram` which plots the system diagram of a diagram (without need for iPython) along with the
  ability to save the SVG file onto your computer.