import faulthandler

faulthandler.enable()

from src.modules.lndf_robot.eval.pose_selector import LocalNDF 
from omegaconf import DictConfig
import os
import os.path as osp
import numpy as np
# from plotly.offline import init_notebook_mode, iplot
import random

os.environ['CKPT_DIR'] = 'src/modules/lndf_robot/ckpts'
seed = 0
# init_notebook_mode()
np.random.seed(seed)
random.seed(seed)