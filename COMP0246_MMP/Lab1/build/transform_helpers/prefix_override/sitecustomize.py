import sys
if sys.prefix == '/home/cody/miniforge3/envs/roboenv-py3.10':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cody/ros2_ws/src/COMP0246_Labs/install/transform_helpers'
