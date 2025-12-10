# code/main.py

"""
Main script for training, validating, and evaluating models for multitemporal pixelwise classification of
satellite image time series.
Handles experiment setup, training loop, validation, checkpointing, and evaluation.
"""

import argparse
import os
import numpy as np
import time
import matplotlib
import rasterio
from tqdm import tqdm
import logging

# Pytorch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.amp import autocast
import torchmetrics

# Own python scripts
from utils import init_weights_const, init_weights_He
from model import MultiTemporal_Model
import functions, args, evaluation
import datamanagement as toker

torch.multiprocessing.set_sharing_strategy('file_system')
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##################### main #################################################

class ExperimentHandler():
    """
    Handles the complete training, validation and evaluation process.
    """

    def __init__(self, **kwargs):
        """
        Initialize ExperimentHandler with configuration parameters.
        Args:
            kwargs: Dictionary of parameters from args.py
        """
        self.P = None                       # Configuration object (args.py)
        self.CM = None                      # Memory object for confusion matrices
        self.top_scores = None              # Dict of best scores for train, val and test
        self.e = None                       # current epoch
        self.F = None                       # dict of folder for saving/loading
        self.net = None                     # model
        self.device = None                  # device to use (GPU or CPU)
        self.n_e = None                     # number of trained epochs
        self.train_start_time = None        # Start time of the whole training process
        self.epoch_start_time = None        # Start time of a epoch
        self.nb_ep_val_notimpr = None       # Number of epochs the validation accuracy did not improve

        self.init_config(kwargs)            # read in all P, create results folder
        self.init_aux_vars()                # Initialize variables that are used during training
        self.init_network_and_device()      # Initialize network variant, check if gpu is available

    def init_config(self, params_dict: dict):
        """
        Configure experiment parameters and create results folder.
        Args:
            params_dict: Dictionary of parameters from args.py
        """
        self.P = argparse.Namespace(**params_dict)

        results_dir = os.path.join(self.P.resu, self.P.name)
        os.makedirs(results_dir, exist_ok=True)
        with open("../config_files/storage_dirs.txt", "r") as f:
            self.P.dast = f.readline().strip() #+ self.P.dast
        self.P.cols, self.P.cls_names = functions.read_csv_DEN(classes_path=self.P.cls_file,return_cls_names=True)

        nbts_map = {
            4: (['0215', '0515', '0815', '1115'], ['0101', '0401', '0701', '1001'], ['0331', '0630', '0930', '1231']),
            6: (['0201', '0401', '0601', '0801', '1001', '1201'],
                ['0101', '0301', '0501', '0701', '0901', '1101'],
                ['0228', '0430', '0630', '0831', '1031', '1231']),
            12: (['0115', '0215', '0315', '0415', '0515', '0615', '0715', '0815', '0915', '1015', '1115', '1215'],
                 ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201'],
                 ['0131', '0228', '0331', '0430', '0531', '0630', '0731', '0831', '0930', '1031', '1130', '1231'])
        }
        if self.P.nbts in nbts_map:
            self.P.t_mid, self.P.t_begin, self.P.t_end = nbts_map[self.P.nbts]

        if self.P.uswe:
            self.list_ious = []
            self.list_ws = []

    def init_aux_vars(self):
        """ Initialization of variables used in the training process.
        """
        self.CM = np.zeros([self.P.nbcl, self.P.nbcl], np.uint32)
        self.top_scores = {'training': 0.0, 'validation': 0.0, 'testing': 0.0}
        self.e = -1
        self.nb_ep_val_notimpr = 0
        base = os.path.join(self.P.resu, self.P.name)
        self.F = {
            'confusion_matrices': os.path.join(base, 'confusion_matrices/'),
            'eval_images': os.path.join(base, 'eval/'),
            'checkpoints': os.path.join(base, 'checkpoints/'),
        }
        for folder in self.F.values():
            if not os.path.exists(folder): os.makedirs(folder)

    def init_network_and_device(self):
        """
        Initialize the network architecture and the device to train on (CPU or GPU).
        """
        logger.info(f'Initializing network for experiment {str(self.P.name)}.')
        self.net = MultiTemporal_Model(img_size=self.P.insz,
                                     patch_size=self.P.ptsz,
                                     in_chans=self.P.nbbd,
                                     embed_dim=self.P.tdim,
                                     dec_head_dim=self.P.swdc,
                                     depths=self.P.swdp,
                                     num_heads=self.P.swhd,
                                     window_size=self.P.swin,
                                     mlp_ratio=self.P.tmlp,
                                     num_classes=len(self.P.cols),
                                     drop_rate=self.P.drop_rate,
                                     attn_drop_rate=self.P.attn_drop_rate,
                                     drop_path_rate=self.P.drop_path_rate,
                                     dropout_ratio=self.P.dropout_ratio,
                                     decoder=self.P.deco,
                                     use_te=self.P.use_te,
                                     tau=self.P.tau,
                                     nbts=self.P.nbts,
                                     ST_light=self.P.ST_light,
                                     patch_embedding=self.P.patch_emb,
                                     conv=self.P.ST_light_conv,
                                     temp_skip=self.P.temp_skip,
                                     nb_CB_PE=self.P.nb_CB_PE,
                                     activation=self.P.conv_activation,
                                     kernel_size=self.P.kernel_size,
                                     pe_skip=self.P.pe_skip
                                    )

        # Parameter initialization
        init_map = {'constant': init_weights_const,'he': init_weights_He,'default': None}
        if self.P.type_init in init_map and init_map[self.P.type_init]:
            self.net.apply(init_map[self.P.type_init])
        elif self.P.type_init != 'default':
            logger.warning("No correct weight initialization chosen!")

        if self.P.cuda == 'cpu' or not torch.cuda.is_available():
            self.device = 'cpu'
            logger.info('Using CPU only!')
        else:
            self.device = f'cuda:{self.P.cuda}'
            logger.info(f'Using GPU number: {self.P.cuda} name: {torch.cuda.get_device_name(int(self.P.cuda))}')

        self.net.to(self.device)
        self.print_n_params()

    def print_n_params(self):
        """
        Count and print the number of parameters and variables of the network.
        """
        params = list(self.net.parameters())
        pp = np.sum([np.prod(list(P.size())) for P in params])
        logger.info(f'Current Model has {pp} parameters in {len(params)} variables')
        self.P.nb_parameters_net = pp
        self.P.nb_varaibles_net = len(params)
        if self.P.mode == 'train': functions.write_parameters_to_txt(self.P)

    def train(self):
        """
        Train the model from scratch or continue training from a checkpoint.
        """
        self.prepare_datasets(train=True, val=True, test=True)
        self.prepare_training_optimizer()
        scheduler = StepLR(self.optimizer, step_size=10, gamma=self.P.lrfa)

        self.class_weights = torch.ones(self.P.nbcl, dtype=torch.float32, device=self.device)
        if self.P.uswe:
            logger.info('Using class weights')
        else:
            logger.info('No class weights used')

        criterion = nn.CrossEntropyLoss(ignore_index=self.P.ignidx, weight=self.class_weights.to(self.device)) if self.P.loss == 'CrEn' else None
        if criterion is None:
            logger.warning('No valid loss function chosen!')

        checkpoint = self.load_checkpoint()
        self.restore_checkpoint(checkpoint)

        val = False
        while self.e < self.P.nepo:
            self.e += 1
            self.print_training_time()
            counter_batches = 0
            CM_ious = self.CM * 0
            for batch in tqdm(self.train_generator):
                counter_batches += 1
                if self.P.use_te:
                    local_batch, local_dates, local_labels = batch['image'], batch['dates'], batch['labels']
                    local_batch, local_dates, local_labels = (
                        local_batch.to(self.device,non_blocking=True),
                        local_dates.to(self.device, non_blocking=True),
                        local_labels.to(self.device, non_blocking=True)
                    )
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                        outputs = self.net((local_batch, local_dates))
                else:
                    local_batch, local_labels = batch['image'], batch['labels']
                    local_batch, local_labels = (
                        local_batch.to(self.device, non_blocking=True),
                        local_labels.to(self.device, non_blocking=True)
                    )
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                        outputs = self.net(local_batch)

                if counter_batches <= self.P.nb_batches_iou and self.P.uswe:
                    preds = torch.argmax(outputs, 1)
                    cm_part = evaluation.calculate_conf_matrix(
                        preds.cpu().data.numpy(),local_labels.cpu().data.numpy(),
                        classes=np.arange(0, self.P.nbcl, 1), ignore_class=self.P.ignidx
                    )
                    CM_ious = CM_ious + cm_part

                if counter_batches == self.P.nb_batches_iou and self.P.uswe:
                    IoUs = evaluation.calc_IoU(CM_ious)
                    self.list_ious.append(IoUs)
                    if self.P.kowe == 'ioua':
                        arr = np.array(self.list_ious[-10:])
                        IoUs = np.mean(arr, axis=0) if arr.shape[0] > 1 else arr[0]
                    meanIoU = np.mean(IoUs)
                    delta_IoUs = IoUs - meanIoU
                    self.class_weights = torch.ones_like(self.class_weights)
                    self.class_weights -= torch.tensor(delta_IoUs, dtype=torch.float32, device=self.device)
                    self.class_weights.pow_(self.P.acep)

                with autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                    loss = criterion(outputs, local_labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                del local_labels, outputs

            logger.info(f'Class weights from current epoch: {self.class_weights.cpu().data.numpy()}')
            if self.e >= self.P.ep_to_val: val = True
            if val:
                self.net.eval()
                self.validate_gpu()
                self.net.train()
            self.save_net('latest.pt')
            scheduler.step()

    def prepare_datasets(self, train: bool, val: bool, test: bool):
        """
        Prepare PyTorch datasets and data loaders for training, validation, and testing.

        Args:
            train (bool): If True a training dataset and generator is created
            val (bool): If True a validation dataset and generator is created.
            test (bool): If True a test dataset and generator is created.
        """
        df_train, df_test, df_val = toker.prepare_df_train_val_test(config=self.P)
        if train:
            self.train_dataset = toker.Train_Dataset_DEN(df=df_train, config=self.P)
            self.train_generator = DataLoader(self.train_dataset, batch_size=self.P.btsz, shuffle=False,
                                                  num_workers=self.P.nbwo, drop_last=True)
        if val:
            self.val_dataset = toker.Eval_Dataset_df(df=df_val, config=self.P)
            self.val_generator = DataLoader(self.val_dataset, batch_size=self.P.btsz, shuffle=False,
                                                num_workers=self.P.nbwo, drop_last=True)
        if test:
            self.test_dataset = toker.Eval_Dataset_df(df=df_test, config=self.P)
            self.test_generator = DataLoader(self.test_dataset, batch_size=self.P.btsz, shuffle=False,
                                                 num_workers=self.P.nbwo, drop_last=True)

    def prepare_training_optimizer(self):
        """
        Prepare the optimizer for training.
        """
        if self.P.optm == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), lr=self.P.lrin,
                momentum=self.P.bta1, weight_decay=self.P.wdec
            )
        elif self.P.optm == 'ADAM':
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=self.P.lrin,
                betas=(self.P.bta1, self.P.bta2), weight_decay=self.P.wdec
            )
        else:
            logger.warning('No implemented optimizer chosen!')

    def load_checkpoint(self, version: str ='latest'):
        """
        Load a checkpoint if available.
        Args:
            version: 'latest' or 'best'
        Returns:
            Loaded checkpoint or None.
        """
        if not torch.cuda.is_available():
            logger.warning('GPU not available, loading checkpoint not implemented.')
            return None

        checkpoint_file = os.path.join(self.F['checkpoints'], 'latest.pt' if version == 'latest' else 'best_val.pt')
        if os.path.exists(checkpoint_file):
            logger.info(f'Loaded checkpoint from {checkpoint_file}')
            return torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        else:
            logger.info('No checkpoint loaded')
            return None

    def restore_checkpoint(self, checkpoint, version: str = 'latest'):
        """
        Restore model and optimizer state from checkpoint.
        Args:
            checkpoint: Loaded checkpoint dictionary.
            version: 'latest' or 'best'
        """
        if checkpoint:
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.e = checkpoint['epoche']
            if version == 'latest':
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.top_scores = checkpoint['top_scores']
                if 'list_ious' in checkpoint:
                    self.list_ious = checkpoint['list_ious']
                    logger.info('Loaded list_ious from checkpoint')
            else:
                self.top_scores = checkpoint['metrics']

    def print_training_time(self):
        """
        Print the time for the current epoch and the total training time.
        """
        if not self.train_start_time:
            self.train_start_time = time.time()
            self.epoch_start_time = time.time()
            self.n_e = -1

        self.n_e += 1
        run_time = time.time() - self.train_start_time
        it_time = (time.time() - self.epoch_start_time)
        self.epoch_start_time = time.time()
        logger.info(f"Epoch {self.e}/{self.P.nepo}, ({it_time:.2f} s/epoch, total: {run_time:.1f} sec)")

    def validate_gpu(self):
        """
        Compute validation accuracy during training.
        """
        dataloader = self.val_generator
        val_acc = torchmetrics.Accuracy(ignore_index=self.P.ignidx, task="multiclass", num_classes=self.P.nbcl).to(self.device)
        prec = torchmetrics.classification.precision_recall.Precision(
            task="multiclass",num_classes=self.P.nbcl, average='none', multidim_average='global', ignore_index=self.P.ignidx
        ).to(self.device)
        rec = torchmetrics.classification.precision_recall.Recall(
            task="multiclass", num_classes=self.P.nbcl, average='none', multidim_average='global', ignore_index=self.P.ignidx
        ).to(self.device)

        for batch in dataloader:
            local_batch, local_labels, start_pos = batch['image'], batch['labels'], batch['startpos']
            local_batch = local_batch.to(self.device)
            local_labels = local_labels.to(self.device)

            with torch.no_grad():
                if self.P.use_te:
                    local_dates = batch['dates'].to(self.device)
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                        output = self.net((local_batch, local_dates))
                else:
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                        output = self.net(local_batch)
                preds = torch.argmax(output, 1)
                val_acc.update(preds, local_labels)
                prec(preds, local_labels)
                rec(preds, local_labels)

        val_acc_final = val_acc.compute()
        prec_final = prec.compute()
        rec_final = rec.compute()
        f1_scores = 2 * (prec_final * rec_final) / (prec_final + rec_final)
        logger.info(f'F1 scores: {f1_scores}')

        # Save current OA, precision, recall values
        file_path = os.path.join(self.F['confusion_matrices'], f'Metrics_val_{self.e}.txt')
        evaluation.save_val_score(
            file_path, self.e, val_acc_final.cpu().data.numpy(),
            prec_final.cpu().data.numpy(), rec_final.cpu().data.numpy()
        )

        score = val_acc_final.cpu().data.numpy()
        if score > self.top_scores['validation']:
            self.top_scores['validation'] = score
            self.save_net('best_val.pt', False)
            self.nb_ep_val_notimpr = 0
        else:
            self.nb_ep_val_notimpr += 1
        if self.nb_ep_val_notimpr == 10:
            self.e = self.P.nepo - 1
        logger.info(f'VALIDATION SCORE/BEST {score:.2%} / {self.top_scores[(name := "validation")]:.2%}')

    def save_net(self, name: str, safe_optimizer: bool = True):
        """
        Save the latest or the best checkpoint (network status).
        Args:
            name: Filename for the checkpoint.
            safe_optimizer: If True, save optimizer state (for latest checkpoint).
        """
        checkpoint_path = os.path.join(self.F['checkpoints'], name)
        if safe_optimizer:
            if 'iou' in self.P.kowe:
                checkpoint = {
                    'epoche': self.e,
                    'top_scores': self.top_scores,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                if 'iou' in self.P.kowe and hasattr(self, 'list_ious'):
                    checkpoint['list_ious'] = self.list_ious
                torch.save(checkpoint, checkpoint_path)
        else:
            torch.save({
                'epoche': self.e,
                'metrics': self.top_scores,
                'model_state_dict': self.net.state_dict()
            }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def eval(self):
        """
        Evaluate the model with test or validation data.
        Loads the trained model and evaluates it on the specified dataset,
        saving results and metrics as needed.
        """
        self.P.btsz = 1
        checkpoint = self.load_checkpoint(version='best')
        self.restore_checkpoint(checkpoint, version='best')
        folder_save = os.path.join(self.F['eval_images'], f"{self.P.mode}/")
        os.makedirs(folder_save, exist_ok=True)

        if 'val' in self.P.mode:
            self.prepare_datasets(train=False, val=True, test=False)
            dataset = self.val_dataset
            dataloader = self.val_generator
        else:
            self.prepare_datasets(train=False, val=False, test=True)
            dataset = self.test_dataset
            dataloader = self.test_generator

        self.net.eval()
        sum_probas, sum_classes, sum_labels = [], [], []

        for _ in range(len(dataset.list_names)):
            sum_probas.append(np.zeros((len(self.P.cols), self.P.nbts, self.P.sat_im_size,self.P.sat_im_size)))
            sum_classes.append(np.zeros((self.P.sat_im_size, self.P.sat_im_size)))
            sum_labels.append((np.zeros((self.P.nbts, self.P.sat_im_size, self.P.sat_im_size))))

        counter = -1
        for batch in tqdm(dataloader):
            local_batch, local_labels, start_pos = batch['image'], batch['labels'], batch['startpos']
            local_batch = local_batch.to(self.device)
            with torch.no_grad():
                if self.P.use_te:
                    local_dates = batch['dates'].to(self.device)
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                        output = self.net((local_batch, local_dates))
                else:
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                        output = self.net(local_batch)
                probas = torch.nn.functional.softmax(output, 1)
            labels = functions.t2n(local_labels[0])
            px = functions.t2n(start_pos[1])[0]
            py = functions.t2n(start_pos[0])[0]

            if px == 0 and py == 0:
                counter += 1
            probas = probas.cpu().data.numpy()
            sum_probas[counter][:, :, px:px + self.P.insz, py:py + self.P.insz] += probas[0]
            sum_classes[counter][px:px + self.P.insz, py:py + self.P.insz] += 1
            sum_labels[counter][:,px:px + self.P.insz, py:py + self.P.insz] = labels

        self.CM *= 0
        self.CM_avg = self.CM
        logger.info('Processing classified images and saving tifs ...')
        for idx, (sum_probas_i, list_labels_i, sum_classes_i) in enumerate(zip(sum_probas, sum_labels, sum_classes)):
            probas_i = sum_probas_i / sum_classes_i
            pred_i = np.argmax(probas_i, 0)

            if 'tif' in self.P.mode:
                if self.P.save_all_timesteps: begin = 0
                else: begin = self.P.nbts - 1
                for ts in range(begin, self.P.nbts):
                    pred_rgb = functions.decode_segmap(pred_i[ts], self.P.cols, axis=0)
                    labels_rgb = functions.decode_segmap(list_labels_i[ts], self.P.cols, axis=0)
                    folder_path = os.path.join(self.P.dast, dataset.list_names[idx][0].lstrip("/"))
                    # Load example tif to have geoloaction
                    tif_to_load = functions.load_first_tif(folder_path)
                    with rasterio.open(tif_to_load) as src:
                        out_prof = src.profile.copy()
                    out_prof["dtype"] = "uint8"

                    month = ts + 1
                    name_tif_rgb = dataset.list_names[idx][2] + str(month) + 'rgb.tif'
                    out_tif_rgb = os.path.join(folder_save, name_tif_rgb)
                    out_prof["count"] = 3
                    with rasterio.open(out_tif_rgb, "w", **out_prof) as dest:
                        dest.write(pred_rgb)

                    name_label_rgb = dataset.list_names[idx][2] + str(month) + 'rgb_label.tif'
                    out_label_rgb = os.path.join(folder_save, name_label_rgb)
                    with rasterio.open(out_label_rgb, "w", **out_prof) as dest:
                        dest.write(labels_rgb)

            conf_mat_part = evaluation.calculate_conf_matrix(
                pred_i, list_labels_i, classes=np.arange(0, self.P.nbcl, 1), ignore_class=self.P.ignidx
            )
            f1_scores, _, _, _, _, _ = evaluation.calc_acc_metrices(
                cm=conf_mat_part, path=None, excel=False,print_cons=False
            )
            self.CM = self.CM + conf_mat_part

        logger.info(f'Accuracy metrics for {self.P.mode} data:')
        evaluation.calc_acc_metrices(
            cm=self.CM, path=os.path.join(folder_save, 'CM_eval_test'), print_cons=True
        )
#################### Functions  ############################################

def main():
    """
        Main entry point for training, validating, and evaluating the model.

        Parses command-line arguments, initializes the experiment handler,
        and runs training and/or evaluation based on the selected mode.
        """
    parser = args.get_parser()
    P = parser.parse_args()
    params_dict = vars(P)

    H = ExperimentHandler(**params_dict)

    if H.P.mode == 'train':
        H.train()
        print('DONE WITH TRAINING, STARTING EVALUATION (TESTING) NOW...')
        H.P.mode = 'test'
        H.eval()
        print('DONE WITH EVAL-TEST, STARTING EVALUATION (VAL) NOW...')
        H.P.mode = 'val'
        H.eval()
    else: H.eval()

if __name__ == "__main__":
    main()