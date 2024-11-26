
# Debug: visualize camera images
import matplotlib.pyplot as plt
from pydrake.all import (
    System,
    Diagram,
    Meshcat,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    PointCloud,
    Rgba
)
import numpy as np
import trimesh
from src.modules.grasping.lndf_robot.utils.plotly_save import plot3d
from plotly.offline import plot
import plotly.graph_objects as go
from manipulation.scenarios import (
    AddMultibodyTriad,
    AddMultibodyPlantSceneGraph,
    Parser,
    ConfigureParser,
)

def visualize_camera_images(station: System):
    context = station.CreateDefaultContext()
    plt.figure(figsize=(12, 6))
    for i in range(3):
        color_image = station.GetOutputPort(f'camera{i}_rgb_image').Eval(context)
        plt.subplot(1, 3, i+1)
        plt.imshow(color_image.data)
        plt.title(f'RGB Image Camera {i}')
    plt.show()

def visualize_depth_images(station: System):
    context = station.CreateDefaultContext()
    plt.figure(figsize=(12, 6))
    for i in range(3):
        color_image = station.GetOutputPort(f'camera{i}_depth_image').Eval(context)
        plt.subplot(1, 3, i+1)
        plt.imshow(color_image.data)
        plt.title(f'Depth Image Camera {i}')
    plt.show()

def visualize_point_cloud(pc: np.ndarray):
    x = pc[0, :]
    y = pc[1, :]
    z = pc[2, :]
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
    fig.show()

def visualize_diagram(diagram: Diagram, output_filename=None):
    import pydot
    import tempfile
    import os
    """
    Utility to visualize the system diagram outside IPython environments.
    Saves the diagram as an SVG file and optionally opens it.

    Args:
        diagram (Diagram): The diagram to visualize.
        output_filename (str): Optional filename to save the SVG. If None, a
                               temporary file is created.
    """
    # Generate Graphviz data from the diagram
    graphviz_data = diagram.GetGraphvizString(max_depth=2)

    # Create a pydot graph from the data
    (graph,) = pydot.graph_from_dot_data(graphviz_data)

    # Define the output file
    if output_filename is None:
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
        output_filename = output_file.name

    # Write the SVG file
    graph.write_svg(output_filename)
    print(f"Diagram saved to {output_filename}")

    # Open the SVG file using the default viewer if desired
    try:
        if os.name == 'posix':  # For macOS and Linux
            os.system(f"open {output_filename}")
        elif os.name == 'nt':  # For Windows
            os.startfile(output_filename)
        else:
            print("Auto-opening SVG file is not supported on this OS.")
    except Exception as e:
        print(f"Could not open SVG file: {e}")

def draw_grasp_candidate(meshcat: Meshcat, X_G, prefix="gripper", draw_frames=True, refresh=False):
    if refresh:
        meshcat.Delete()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    gripper = parser.AddModelsFromUrl(
        "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
    )
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), X_G)
    if draw_frames:
        AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)
    plant.Finalize()

    params = MeshcatVisualizerParams()
    params.prefix = prefix
    meshcat_vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

def draw_query_pts(meshcat: Meshcat, query_pts: np.ndarray) -> None:
    '''
    Draws a set of query points in meshcat.
    '''
    cloud = PointCloud(query_pts.shape[0])
    cloud.mutable_xyzs()[:] = query_pts.T
    meshcat.SetObject('query_pts', cloud, point_size=0.01, rgba=Rgba(1.0, 0, 0))