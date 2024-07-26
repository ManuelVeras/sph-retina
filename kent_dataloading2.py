import copy
import os
import os.path as osp
import warnings
import pdb
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

def load_config(config_path, cfg_options=None):
    """
    Load and process the configuration file.

    Args:
        config_path (str): Path to the configuration file.
        cfg_options (dict, optional): Additional configuration options to merge.

    Returns:
        Config: Processed configuration object.
    """
    cfg = Config.fromfile(config_path)

    # Replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # Update data root according to MMDET_DATASETS
    update_data_root(cfg)

    # Merge additional configuration options if provided
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    return cfg

def build_datasets(cfg):
    """
    Build datasets based on the configuration.

    Args:
        cfg (Config): Configuration object.

    Returns:
        list: List of dataset objects.
    """
    datasets = [build_dataset(cfg.data.train)]
    
    # Check if validation is part of the workflow and add validation dataset
    if len(cfg.workflow) == 2:
        assert 'val' in [mode for (mode, _) in cfg.workflow]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.get(
            'pipeline', cfg.data.train.dataset.get('pipeline'))
        datasets.append(build_dataset(val_dataset))
    
    return datasets

def build_and_print_dataloader(cfg):
    """
    Build dataloaders for the datasets and print the first few batches.

    Args:
        cfg (Config): Configuration object.
    """
    datasets = build_datasets(cfg)
    
    # Iterate over datasets and build dataloaders
    for dataset in datasets:
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False
        )
        
        # Print the first 5 batches of data
        for i, data in enumerate(dataloader):
            pdb.set_trace()
            if i >= 5:
                break
            print(f'Batch {i}: {data}')

def train_model(cfg):
    """
    Train the model based on the configuration.

    Args:
        cfg (Config): Configuration object.
    """
    # Initialize the model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # Build datasets
    datasets = build_datasets(cfg)

    # Set up the training environment
    setup_multi_processes(cfg)
    cfg.device = get_device()
    #seed = init_random_seed(cfg.seed, device=cfg.device)
    #set_random_seed(seed, deterministic=cfg.deterministic)
    #cfg.seed = seed

    # Initialize the logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join('work_dir', f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Log environment info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    # Train the model
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        meta=dict(env_info=env_info,exp_name=osp.basename('work_dir'))
    )

def main(config_path, cfg_options=None):
    """
    Main function to load configuration and train the model.

    Args:
        config_path (str): Path to the configuration file.
        cfg_options (dict, optional): Additional configuration options to merge.
    """
    cfg = load_config(config_path, cfg_options)
    train_model(cfg)

if __name__ == '__main__':
    # Directly specify the config path and options here
    config_path = 'configs/retinanet/sph_retinanet_r50_fpn_120e_indoor360.py'
    cfg_options = {
        'key1': 'value1',
        'key2': 'value2',
        # Add more key-value pairs as needed
    }

    main(config_path, cfg_options)