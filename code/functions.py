# code/functions.py

"""
General utility functions for the training process.
Includes PyTorch tensor conversions, segmentation decoding, parameter logging, and class color reading.
"""

import torch
import pandas as pd
from pylab import *
import os
from datetime import datetime
from typing import Any, Dict, List

######### Pytorch specific functions #####################################################

def t2n(t: torch.Tensor): return t.cpu().data.numpy()

def n2t(n: np.ndarray): return torch.from_numpy(n)

################ general reading/writing functions used during training process ###########

def load_first_tif(folder_path):
    """Load the first .tif file from folder_path and return its array."""
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.tif'):
            return os.path.join(folder_path, fname)
    raise FileNotFoundError("No .tif file found in {}".format(folder_path))

def decode_segmap(image: np.ndarray, cols: Dict[int, List[int]], axis: int = 2):
    """
    Produce an RGB image based on class colors.
    Args:
        image: 2D numpy array (height, width) with class indices.
        cols: Dictionary mapping class index to RGB color list.
        axis: Axis for stacking RGB channels.
    Returns:
        RGB image as a numpy array.
    """
    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    for l in range(0, len(cols)):
        idx = image == l
        r[idx] = cols[l + 1][0]
        g[idx] = cols[l + 1][1]
        b[idx] = cols[l + 1][2]

    rgb = np.stack([r, g, b], axis=axis)
    return rgb

def write_parameters_to_txt(Parameters: Any):
    """
    Write all parameters from the args object to a text file in the results folder.
    Args:
        Parameters: Object containing experiment parameters.
    """
    dic = vars(Parameters)
    output_dir = os.path.join(Parameters.resu, Parameters.name)
    os.makedirs(output_dir, exist_ok=True)
    args_path = os.path.join(output_dir, 'args.txt')
    with open(args_path, 'w') as txt_file:
        for key in dic:
            txt_file.write(f"{key}: {dic[key]}\n")
    save_command_line_args(output_dir)

def save_command_line_args(output_dir: str):
    """
        Save the command line arguments and timestamp to a file.
        Args:
            output_dir: Directory to save the command.txt file.
        """
    os.makedirs(output_dir, exist_ok=True)
    command_path = os.path.join(output_dir, 'command.txt')
    # Save the command line arguments to a file
    with open(command_path, 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('Timestamp: ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')

def read_csv_DEN(classes_path: str ='classes.csv', return_cls_names: bool = False):
    """
    Read the CSV file for class structure and colors.
    Args:
        classes_path: Path to the CSV file with class names and colors.
        return_cls_names: If True, also return class names.
    Returns:
        dict_class_colors: Dictionary mapping class index to RGB color list.
        class_names (optional): Array of class names.
    """
    classes = pd.read_csv(classes_path)
    class_names = classes['Class_Name'].values
    dict_class_colors = {}
    for i in range(len(classes)):
        dict_class_colors[i + 1] = [
            classes['rot'].values[i],
            classes['grün'].values[i],
            classes['blau'].values[i]
        ]
    if return_cls_names:
        return dict_class_colors, class_names
    else: return dict_class_colors