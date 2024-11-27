from manipulation.create_sdf_from_mesh import create_sdf_from_mesh
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='wooden_basket')
    parser.add_argument('-scale', type=float, default=2)
    parser.add_argument('-mass', type=float, default=0.5)
    args = parser.parse_args()

    create_sdf_from_mesh(Path(f'src/assets/{args.name}/{args.name}.obj'), args.mass, args.scale, True, 5e7, 1.25, 0.8, 1.2, False)