from pathlib import Path
import argparse
import yaml
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model import MLCGCN, Model
from train import MLCTrain, MutiMLCTrain
from transformers import get_linear_schedule_with_warmup
from datetime import datetime
from dataloader import init_dataloader


def main(args, dataseed):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    dataloaders, node_size, node_feature_size, timeseries_size, out_size = init_dataloader(config['data'], dataseed)
    num_train_batch = len(dataloaders[0])
    config['train']["seq_len"] = timeseries_size
    config['train']["node_size"] = node_size
    config['model']['roi_num'] = node_size
    config['model']['node_feature_dim'] = node_feature_size
    config['model']['time_series'] = timeseries_size
    config['model']['out_size'] = out_size
    for dataloader in dataloaders:
        model = Model(config['model'])
        if out_size > 2:
            use_train = MutiMLCTrain
        else:
            use_train = MLCTrain

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'])

        warm_up = config['train']['warm_up']
        if warm_up:
            total_steps = num_train_batch * config['train']['epochs']
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * config['train']['warm_up_step'],
                                                        num_training_steps=total_steps)
        else:
            scheduler = None

        loss_name = 'loss'
        if config['train']["sparsity_loss"]:
            loss_name = f"{loss_name}_sparsity_loss"
        if config['train']["group_loss"]:
            loss_name = f"{loss_name}_group_loss"

        embedding_size = config['model']['embedding_size'] if 'embedding_size' in config['model'] else "none"
        window_size = config['model']['window_size'] if 'window_size' in config['model'] else "none"
        num_trans_layers = config['model']['num_trans_layers'] if 'num_trans_layers' in config['model'] else "none"

        save_folder_name = Path(config['train']['log_folder'])/Path(
            f"_{config['data']['dataset']}_{config['model']['type']}_{config['train']['method']}"
            + f"_{loss_name}_{embedding_size}_{window_size}_{num_trans_layers}")

        #Start Training
        train_process = use_train(config, model, optimizer, scheduler, dataloader, save_folder_name)
        train_process.train()


seed = 0
dataseed = 0
torch.manual_seed(seed)
np.random.seed(seed)

#Args Read
parser = argparse.ArgumentParser(description='MLC-GCN')
parser.add_argument('--config_filename', default='setting/config1.yaml', type=str,
                    help='Configuration filename for training the model.')

args = parser.parse_args()
main(args, dataseed)
