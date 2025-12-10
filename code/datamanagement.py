# code/datamanagement.py

"""
Data management utilities for Dynamic Earth dataset.
Includes dataset classes, patch loading, augmentation, and dataframe preparation.
"""

from pathlib import Path
from typing import Any
import numpy as np
import rasterio
from rasterio.windows import Window
from numpy.random import randint
import torch
from torch.utils import data
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################ Image Augmentation ###################################

def augment(image: np.ndarray, label: np.ndarray, augmentation: str, flip: int, rot: int):
    """
    Perform image augmentation on the input image and label.

    Args:
        image: Input image array.
        label: Corresponding label array.
        augmentation: Type of augmentation ('trans' or 'flip').
        flip: Flip code (0: none, 1: vertical, 2: horizontal, 3: both).
        rot: Number of 90-degree rotations.

    Returns:
        Tuple of augmented image and label.
    """
    if augmentation == 'trans': return image, label
    elif augmentation == 'flip':
        if flip in (1,3):  # vertical flip
            image = np.flip(image, axis=2)
            label = np.flip(label, axis=1)
        if flip in (2,3):  # horizontal flip
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=0)
        image = image.copy()
        label = label.copy()
        # random rotarion about 0, 90, 280, 270 degree
        image = np.rot90(image, k=rot, axes=(1, 2))
        label = np.rot90(label, k=rot, axes=(0, 1))
        image = image.copy()
        label = label.copy()
        return image, label
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation}")

############################ Datasets #############################################

def select_dayinmonth(cur_month: str):
    """
        Select a random day as a string for the given month.
        Args:
            cur_month: Month as a string, e.g. '01'.
        Returns:
            Day as a zero-padded string, e.g. '01'.
        """
    if cur_month in {'01', '03', '05', '07', '08', '10', '12'}:
        day = randint(1, 32)
    elif cur_month == '02':  # 28 days
        day = randint(1, 29)
    else:  # 30 days
        day = randint(1, 31)
    if day < 10:
        day = '0' + str(day)
    else:
        day = str(day)
    return day

class Train_Dataset_DEN(data.Dataset):
    '''
    PyTorch Dataset for Dynamic Earth dataset by Toker et al. (2021).

    Args:
        df: DataFrame with image metadata.
        config: Configuration object.

    Returns:
        len(): returns the length of the training dataset (here: images per epoch)
        getitem(index): returns one sample of data from the dataset
                    returns X, Y, [dates], [aoi]
                    X: pytorch tensor, image data (n_channels, insz, insz)
                    Y: pytorch tensor, label data (insz, insz)
                    dates: pytorch tensor with dates of the time series (only if config.use_te==True)
                    aoi: list of strings with aoi names (only if config.use_te==True)
    '''

    def __init__(self, df: pd.DataFrame, config: Any):
        self.config = config
        self.df_train = df
        self.insz = self.config.insz
        self.list_groupnames = []
        self.df_train_groups = self.df_train.groupby(['path_planet', 'year'])
        for name, group in self.df_train_groups:
            self.list_groupnames.append(name)

    def __len__(self):
        return self.config.impe

    def __getitem__(self, index: int):
        rng_tile = np.random.default_rng()
        rng_x = np.random.default_rng()
        x_start = rng_x.integers(self.config.sat_im_size - self.insz)
        if x_start == 0: x_start = 1
        rng_y = np.random.default_rng()
        y_start = rng_y.integers(self.config.sat_im_size - self.insz)
        if y_start == 0: y_start = 1

        random_aoi = rng_tile.integers(len(self.list_groupnames)) # random aoi + random year
        cur_group_name = self.list_groupnames[random_aoi]
        cur_group = self.df_train_groups.get_group(cur_group_name)
        for t in range(len(cur_group.index)):
            if self.config.mets == 'periods_random': day = select_dayinmonth(cur_group.date_str.iloc[t][-2:])
            elif self.config.mets == 'periods_first': day = '01'
            elif self.config.mets == 'periods_middle': day = '15'
            cur_group.loc[cur_group.index[t], 'date_str'] = cur_group.loc[cur_group.index[t], 'date_str'][:] + '-' + day

        label = np.empty(shape=(self.config.nbts, self.insz, self.insz))
        image = np.empty(shape=(self.config.nbbd, self.config.nbts, self.insz,self.insz))
        for i in range(len(cur_group.index)): # iterate over the timesteps we want to load one after the other
            aoi = cur_group.iloc[i]
            image_i, label_i = load_patch_df(self.config, aoi, self.insz, x_start, y_start)
            if self.config.onau != 'trans':
                onau_flip = randint(0, 3)
                onau_rot = randint(0, 3)
                image_i, label_i = augment(image_i, label_i, self.config.onau, onau_flip, onau_rot)
            label[i,:,:] = label_i
            image[:,i,:,:] = image_i

        if self.config.use_te:
            dates_list = [d.replace('-', '') for d in cur_group.date_str.tolist()]
            dates_list = [int(d[:]) for d in dates_list]
            aoi_names = [x for x in cur_group.path_planet.tolist()]
            return {'image': torch.from_numpy(image).float(), 'labels': torch.from_numpy(label).long(),
                    'dates': torch.Tensor(dates_list).long(), 'aoi': aoi_names}
        else:
            return {'image': torch.from_numpy(image).float(), 'labels': torch.from_numpy(label).long()}

class Eval_Dataset_df(data.Dataset):
    '''
    Characterizes a dataset for PyTorch
    Dataset for evaluation with Dynamic Earth dataset by Toker et al. (2021).
    Args:
        df: DataFrame with image metadata.
        config: Configurations as dictionary (from args.py)
    Returns:
        len(): Returns number of patches to classify for testing
        getitem(index): Dictionary with image, labels, startpos, and optionally dates used to be classified.
                    returns dict{'image','labels','startpos', ['dates']}
                    'image': pytorch tensor, image data (nbbd, nbts, insz, insz)
                    'labels': pytorch tensor, label data (nbts, insz, insz)
                    'startpos': pytorch tensor with 3 entries: number of planet tile, x_start, y_start
                    ['dates']: pytorch tensor with dates of the time series (only if config.use_te==True)
    '''
    def __init__(self, df: pd.DataFrame, config: Any):
        self.config = config
        self.df_eval = df.sort_values(by=['path_planet', 'date_str', 'x', 'y'])
        self.list_names = []
        self.df_eval_groups = self.df_eval.groupby(['path_planet', 'year'])
        for path, _ in self.df_eval_groups:
            # path: tuple (path_planet, year), e.g. ('/planet/44N/15E-32N/5912_3937_13/PF-SR/', '2018')
            list_path_aoi = str(path[0]).split('/')
            name_to_save = list_path_aoi[2] + '_' + list_path_aoi[3] + '_' + list_path_aoi[4] + '_' + str(path[1])
            # add name_to_save to path tuple:
            new_list = (path[0], path[1], name_to_save)
            self.list_names.append(new_list)

        self.list_groupnames_xy = []
        self.df_eval_groups_xy = self.df_eval.groupby(['path_planet', 'year', 'x', 'y'])
        for name, group in self.df_eval_groups_xy:
            self.list_groupnames_xy.append(name)

    def __len__(self):
        return len(self.list_groupnames_xy)

    def __getitem__(self, index: int):
        cur_group_name = self.list_groupnames_xy[index]
        cur_group = self.df_eval_groups_xy.get_group(cur_group_name).copy()
        for t in range(len(cur_group.index)):
            cur_group.loc[cur_group.index[t], 'date_str'] = cur_group.loc[cur_group.index[t], 'date_str'][:] + '-'+'01'

        startpos = [cur_group_name[2], cur_group_name[3]]
        if self.config.use_te:
            dates_list = [d.replace('-', '') for d in cur_group.date_str.tolist()]
            dates_list = [int(d[:]) for d in dates_list]
        image = np.empty(shape=(self.config.nbbd, self.config.nbts, self.config.insz, self.config.insz))
        label = np.empty(shape=(self.config.nbts, self.config.insz, self.config.insz))
        for i in range(len(cur_group.index)):
            aoi = cur_group.iloc[i]
            image_i, label_i = load_patch_df(self.config, aoi, self.config.insz, startpos[0], startpos[1])
            label[i,:,:] = label_i
            image[:, i, :, :] = image_i

        if self.config.use_te:
            return {'image': torch.from_numpy(image).float(),
                    'labels': torch.from_numpy(label).long(),
                    'dates': torch.Tensor(dates_list).long(),
                    'startpos': startpos}
        else:
            return {'image': torch.from_numpy(image).float(),
                    'labels': torch.from_numpy(label).long(),
                    'startpos': startpos}

########################### Data loading ##########################################

def load_patch_df(config: Any, aoi: Any, im_size: int, x_start: int, y_start: int):
    """
        Load a patch of image and label data.
        Args:
            config: Configuration object.
            aoi: Area of interest row.
            im_size: Patch size.
            x_start: X offset.
            y_start: Y offset.
        Returns:
            Tuple of (image channels, label array).
        """
    window = Window(x_start, y_start, im_size, im_size)
    file_to_load_label = Path(config.dast + aoi.path_labels)
    db_label = rasterio.open(file_to_load_label)
    label = db_label.read(window=window)
    label = np.argmax(label, axis=0)
    filename = '/' + aoi.date_str + '.tif'
    file_to_load = Path(config.dast + aoi.path_planet + filename)
    if not file_to_load.is_file():
        day = int(str(file_to_load)[-6:-4])-1
        file_to_load = Path(str(file_to_load)[:-6] + f"{day:02d}.tif")

    with rasterio.open(file_to_load) as src:
        channels = src.read(window=window)
        channels[channels > 10000] = 10000
        channels = normalize_planet(channels)
    return channels, label

def normalize_planet(channel):
    """
    Normalize all input channels.
    Args:
        channel: Input array (bands, H, W).
    Returns:
        Normalized array.
    """
    mean = [1042.59, 915.62, 671.26, 2605.21]
    std = [957.96, 715.55, 596.94, 1059.90]
    nb_bands = channel.shape[0]
    for i in range(nb_bands):
        channel[i,:,:] = (channel[i,:,:] - mean[i]) / (std[i] + 1e-5)
    return channel

########################## Dataframe utilities ####################################

def add_xy_offsets(df_val: pd.DataFrame, config: Any):
    """
    Add x and y offsets to the validation/test dataframe.
    Args:
        df_val: Input dataframe.
        config: Configuration object.
    Returns:
        Dataframe with x/y offsets.
    """
    h, w = config.sat_im_size, config.sat_im_size
    y = 0
    df_val_xy = None
    while True:
        x = 0
        while True:
            df_val["x"] = x
            df_val["y"] = y
            if x == 0 and y == 0: df_val_xy = df_val.copy()
            else: df_val_xy = pd.concat([df_val_xy, df_val], ignore_index=True)
            if x == h - config.insz: break
            x += int(config.insz/2)
            if x + config.insz >= h: x = h - config.insz
            if x % 2 != 0: x = x - 1
        if y == w - config.insz: break
        y += int(config.insz/2)
        if y + config.insz >= w: y = w - config.insz
        if y % 2 != 0: y = y - 1
    return df_val_xy

def prepare_df_train_val_test(config: Any):
    """
    Prepare dataframes for training, validation, and testing with the Dynamic Earth dataset.
    Args:
        config: Configuration object.
    Returns:
        Tuple of (train_df, test_df, val_df).
    """

    df_train = pd.read_csv(Path(config.dast + config.list_train + '.csv'))
    df_val = pd.read_csv(Path(config.dast + config.list_val + '.csv'))
    df_test = pd.read_csv(Path(config.dast + config.list_test + '.csv'))

    df_train["date"] = pd.to_datetime(df_train["date_str"], format='%Y-%m')
    df_val["date"] = pd.to_datetime(df_val["date_str"], format='%Y-%m')
    df_test["date"] = pd.to_datetime(df_test["date_str"], format='%Y-%m')

    df_train["year"] = df_train.date.dt.strftime("%Y")
    df_val["year"] = df_val.date.dt.strftime("%Y")
    df_test["year"] = df_test.date.dt.strftime("%Y")

    df_val = add_xy_offsets(df_val, config)
    df_test = add_xy_offsets(df_test, config)

    return df_train, df_test, df_val