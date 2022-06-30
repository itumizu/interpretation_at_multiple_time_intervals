import os
import re
import random
# import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_name="./configs/XGBoost.yml")
def makeConfigs(cfg: DictConfig):
    hyper_params = cfg.trainer.hyper_params

    current_dir = re.sub("/configs.+", "", os.getcwd())
    print(current_dir)

    seed = hyper_params.seed
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    pl.seed_everything(seed)

    is_regression = True

    data_path = f"{current_dir}/data/hba1c/included/1_year_interval"
    # data_path = f"{current_dir}/data/hba1c/not_included/1_year_interval"
    # data_path = f"{current_dir}/data/creatinine/included/1_year_interval"
    # data_path = f"{current_dir}/data/creatinine/not_included/1_year_interval"
    
    current_date = datetime.now()
    current_day = current_date.strftime('%Y-%m-%d') 
    current_time = current_date.strftime('%H-%M-%S')

    conditions = []

    for is_used_raw_data in hyper_params.is_used_raw_data:
        condition = {
            'is_used_raw_data': is_used_raw_data
        }

        conditions.append(condition)

    # print(conditions)
    
    metrics = "RMSE"

    if metrics == "RMSE":
        optimize_direction = "minimize"
        
    else:
        optimize_direction = "maximize"
    
    for index, params in enumerate(conditions):
        config = OmegaConf.create(OmegaConf.to_yaml(cfg))
        config_hydra = OmegaConf.create({
            "hydra":{
                "run": {
                    "dir":  current_dir + "/outputs/XGBoost/${name}/"
                }
            }
        })
        
        file_name = ["_".join([str(key), str(value)]) for key, value in params.items()]

        config.trainer.hyper_params.is_used_raw_data = params['is_used_raw_data']

        gpu_count = 0

        # 各フォールドごとに行う
        for fold_number in range(0, 5):
            config.trainer.data_path = data_path
            config.trainer.gpu = gpu_count
            config.trainer.debug = False
            config.trainer.config_name = "_".join(file_name) + f"_Fold_{fold_number}" + ".yml"

            config.trainer.hyper_params.optimize_direction = optimize_direction
            config.trainer.hyper_params.metrics = metrics
            config.trainer.hyper_params.is_regression = is_regression
            
            config.trainer.hyper_params.fold_number = fold_number
            config.trainer.hyper_params.output_path = f"{current_dir}/results/XGBoost/{current_day}/{current_time}/{'_'.join(file_name)}/Fold_{fold_number}"
            config.trainer.output_path = f"{current_dir}/results/XGBoost/{current_day}/{current_time}/{'_'.join(file_name)}/Fold_{fold_number}"

            config.name = "_".join(file_name) + f"_Fold_{fold_number}"

            config = OmegaConf.merge(config, config_hydra)

            output_path = Path("_".join(file_name), f"{fold_number}")

            os.makedirs(output_path, exist_ok=True)
            os.makedirs(config.trainer.output_path, exist_ok=True)

            with open(Path(output_path, "_".join(file_name) + ".yml"), mode="w", encoding='utf_8_sig') as output_file:
                output_file.write(OmegaConf.to_yaml(config))
            
            gpu_count += 1

            if gpu_count > 3:
                gpu_count = 0


if __name__ == '__main__':
    makeConfigs()