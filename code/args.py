# code/args.py

"""
Argument parser for Dynamic Earth Net experiments.
Defines all command-line arguments for training, evaluation, and data management.
"""

import os
import argparse

def add_arguments(parser: argparse.ArgumentParser):
    """
        Add all command-line arguments to the parser.
        Args:
            parser: An argparse.ArgumentParser instance.
    """
    # Experiment setup
    parser.add_argument("--name", type=str, help="Experiment name", default='test_1')
    #parser.add_argument("--dast", type=str, help="path to data folder",
    #                    default='DynamicEarthNet/storage_dyn_earth_net/')
    parser.add_argument("--resu", type=str, help="path to results", default='./results/')  # default='./results/')
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'test', 'tif_test', 'tif_val'], default='train')
    parser.add_argument("--cuda", type=str, help="CUDA device to use",default=os.environ.get("CUDA_VISIBLE_DEVICES","0"))

    # Data --> here for the Dynamic Earth dataset, adapt if needed for different datasets
    parser.add_argument("--sat_im_size", type=int, help="Height and width of sat. images", default=1024)
    parser.add_argument("--list_train", type=str, help="name of csv list to train", default='train')
    parser.add_argument("--list_val", type=str, help="name of csv list to valid", default='val')
    parser.add_argument("--list_test", type=str, help="name of csv list to test", default='test')
    parser.add_argument("--nbbd", type=int, help='Nb. bands to use, r.g. rgb -> 3', default=4)
    parser.add_argument("--nbcl", type=int, help="nb of classes", choices=[2,3,4,6,7,8,9], default=6)
    parser.add_argument("--cls_file", type=str, help="File that contains colors for classes",
                        default='../example/classes_Planet.csv')
    parser.add_argument("--soti", nargs='+', help='start and end date', default=['2018-01-01', '2019-12-31'])
    parser.add_argument("--ignidx", type=int, help="Index of class to be ignored in train/eval process", default=6)
    parser.add_argument("--save_all_timesteps", type=bool, help="True: Save pred. (tifs) for all ts, "
                                                                "False: Save only for last ts", default=False)

    # Model architecture
    parser.add_argument("--nbts", type=int, help="Number of in & out time steps", choices=[1,4,6,12], default=12)
    parser.add_argument("--insz", type=int, help="size of input patches (square)", default=256)
    parser.add_argument("--modl", type=str, choices=[ 'swin', 'swin_mt'],help="architecture", default='swin_mt')
    parser.add_argument("--deco", type=str, choices=["upernet"], help="decoder", default="upernet")
    parser.add_argument("--patch_emb", type=str, help="swin-wie im swin, conv2d-2dconv+downsampl separat für T, "
                                        "conv3d-3dconv. to already get spatial-temp. features", default="conv3d")
    parser.add_argument("--nb_CB_PE", type=int, help="nb of conv layers for patch emb", default=2)
    parser.add_argument("--pe_skip", type=bool, help="Skip connections at patch emb. stage", default=False) # todo: vielleicht raus?
    parser.add_argument("--ptsz", type=int, help="transformer patch size", default=4)
    parser.add_argument("--tdim", type=int, help="transformer latent vector size", default=48)
    parser.add_argument("--swdp", nargs='+', type=int, help="Nb. layers in diff. stages", default=[2,2])
    parser.add_argument("--swhd", nargs='+', type=int, help="Nb. parallel MSA-Heads in diff. stages", default=[3,6])
    parser.add_argument("--swin", type=int, help="SwinTransformer window size for local self-attention", default=7)
    parser.add_argument("--tmlp", type=int, help="transformer dimension of mlp (factor of tdim)", default=4)
    parser.add_argument("--swdc", type=int, help="SwinTransformer decoder internal dimension", default=512)
    parser.add_argument("--use_te", type=bool, help="Use temp. enc. or not. Yes --> dataloader provides tuple of image "
                                                    "batch and dates", default=True)
    parser.add_argument("--tau", type=int, help="tau for te with DOY", default=10000)
    parser.add_argument("--ST_light", type=bool, help="To use (1) light ST-TB or not (0)", default=True)
    parser.add_argument("--ST_light_conv", type=str, help=" 'a'-attention in sp. dim,'c'-conv in sp. dim.", default='c')
    parser.add_argument("--temp_skip", type=bool,help="Use temp weight. in skip conn.(True) or not(False)",default=False)
    parser.add_argument("--drop_rate", type=float, help="dropout nach PE+TE", default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, help="dropout einzelner Attent. in A=softm(QK..)", default=0.4)
    parser.add_argument("--drop_path_rate", type=float, help="stochastic depth dropout", default=0.2)
    parser.add_argument("--dropout_ratio", type=float, help="dropout UPerNet", default=0.1)
    parser.add_argument("--conv_activation", type=str, help="activation function for all conv. involved layers in swin",
                        choices=['relu', 'leakyrelu', 'gelu'], default='leakyrelu')
    parser.add_argument("--kernel_size", type=int, help="kernel size for conv layers", default=3)
    parser.add_argument("--type_init", type=str, help="type of par. initial.", choices=['constant', 'he', 'default'],
                        default='default')

    # Training
    parser.add_argument("--ep_to_val", type=int, help="nb of epochs until val. starts", default=30)
    parser.add_argument("--nepo", type=int, help="max number of epochs", default=100)
    parser.add_argument("--impe", type=int, help="images (patches) per epoche", default=2000)
    parser.add_argument("--btsz", type=int, help="batch size", default=2)
    parser.add_argument("--nbwo", type=int, help="nb of workers in parallel for data loading", default=64)
    parser.add_argument("--mets", type=str, help="Method for creating time series",
                        choices=['periods_random', 'periods_middle', 'periods_first'], default='periods_random')

    parser.add_argument("--loss", type=str, choices=['CrEn', 'FoLo'], help="Which loss to use", default='CrEn')
    parser.add_argument("--uswe", type=bool, help="Whether to use cls weights (true) or not (False)", default=True)
    parser.add_argument("--kowe", type=str, help="kind of weights", choices=['iou', 'ioua'], default='ioua')
    parser.add_argument("--acep", type=int, help="parameter k in iou formular, influence of IoU on weights", default=1)
    parser.add_argument("--nb_batches_iou", type=int, help="", default=100)

    parser.add_argument("--optm", type=str, choices=['SGD', 'ADAM'], default='ADAM')
    parser.add_argument("--lrin", type=float, help="Initial learning rate", default=6e-05)
    parser.add_argument("--lrfa", type=float, help="factor which decreases the l_r every 10 epochs", default=0.7)
    parser.add_argument("--bta1", type=float, help="mom zero -> SGD/ADAM", default=0.9)
    parser.add_argument("--bta2", type=float, help="zero -> SGD/MOM", default=0.999)
    parser.add_argument("--wdec", type=float, help="weight decay zb 1e-10", default=0)
    parser.add_argument("--onau", type=str, choices=['trans', 'flip'], help="Type of data augmentation"
                        "trans: just rand translation, flip: + random flip and 90degree rot, ", default='flip')

    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

def get_parser():
    """
     Create and return an argument parser with all arguments added.

    Returns:
        An argparse.ArgumentParser instance with all arguments.
    """
    parser = argparse.ArgumentParser(description="Dynamic Earth Net argument parser")
    add_arguments(parser)
    return parser