import argparse
from pydrake.all import (
    Meshcat, StartMeshcat, Simulator
)
from omegaconf import DictConfig
from src.setup import build_pouring_diagram
import logging

class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")

def pouring_demo(cfg: DictConfig, meshcat: Meshcat):
    meshcat.Delete()
    diagram, plant = build_pouring_diagram(meshcat, cfg)
    simulator = Simulator(diagram)

    simulator.AdvanceTo(0.1)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    meshcat.StartRecording()

    # run as fast as possible
    simulator.set_target_realtime_rate(0)
    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if simulator.get_context().get_time() > cfg.max_time:
            raise Exception("Took too long")
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        # stats = diagram.get_output_port().Eval(simulator.get_context())
        meshcat.StopRecording()
    meshcat.PublishRecording()
    meshcat.DeleteButton("Stop Simulation")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Local NDF Pouring Robot'
    )
    parser.add_argument('-n', help='Number of balls to add to box', default=20)
    parser.add_argument('-m', help='Basket ID to use', default=1)

    args = parser.parse_args()
    cfg = DictConfig({
        'num_balls': args.n,
        'basket_id': args.m,
        'max_time': 10.0,
    })

    logging.getLogger('drake').addFilter(NoDiffIKWarnings())

    meshcat = StartMeshcat()
    pouring_demo(cfg, meshcat)
