from manipulation.create_sdf_from_mesh import create_sdf_from_mesh
from pathlib import Path

if __name__ == '__main__':
    create_sdf_from_mesh(Path('src/assets/bowl/bowl.obj'),1, 1, True, 5e7, 1.25, 0.8, 1.2, False)