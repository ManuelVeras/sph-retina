import copy
import os
import os.path as osp
import warnings
import pdb

import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.utils import replace_cfg_vals, update_data_root

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

def main(config_path, cfg_options=None):
    """
    Main function to load configuration and build dataloaders.

    Args:
        config_path (str): Path to the configuration file.
        cfg_options (dict, optional): Additional configuration options to merge.
    """
    cfg = load_config(config_path, cfg_options)
    build_and_print_dataloader(cfg)

if __name__ == '__main__':
    # Directly specify the config path and options here
    config_path = 'kent_configs/retinanet/kent_retinanet_r50_fpn_120e_indoor360.py'
    cfg_options = {
        'key1': 'value1',
        'key2': 'value2',
        # Add more key-value pairs as needed
    }

    main(config_path, cfg_options)