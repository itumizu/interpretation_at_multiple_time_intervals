import re
import os
import copy
import glob
import json
import random
import sqlite3
import logging
import pickle
from pathlib import Path
from datetime import datetime

# import requests

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from sklearn.utils import class_weight

import hydra
from hydra import utils
from omegaconf import DictConfig, OmegaConf

import optuna
from optuna.samplers import TPESampler

from src.methods import XGBOOST

def output(
        output_path: Path,
        hyper_params, 
        results: dict) -> None:

    path = Path(hyper_params.output_path, output_path)
    results['hyper_params'] = OmegaConf.to_container(hyper_params, resolve=True)

    os.makedirs(path, exist_ok=True)

    with open(Path(path, "result.json"), 'w') as f:
        json.dump(results, f, indent=4)

def objective_with_data(
            train_data_list,
            train_labels_list,
            val_data_list,
            val_labels_list,
            test_data_list,
            test_labels_list,
            data_unlabeled_list,
            class_weights_list,
            categorical_columns_list,
            hyper_params):

    params = copy.deepcopy(hyper_params)

    def objective(trial):
        eta = trial.suggest_categorical('eta', params.eta)
        min_child_weight = trial.suggest_categorical("min_child_weight", params.min_child_weight)
        gamma = trial.suggest_categorical("gamma", params.gamma)
        colsample_bytree = trial.suggest_categorical("colsample_bytree", params.colsample_bytree)
        subsample = trial.suggest_categorical("subsample", params.subsample)
        max_depth = trial.suggest_categorical("max_depth", params.max_depth)

        hyper_params.eta = eta
        hyper_params.max_depth = max_depth
        hyper_params.min_child_weight = min_child_weight
        hyper_params.gamma = gamma
        hyper_params.colsample_bytree = colsample_bytree
        hyper_params.subsample = subsample

        logging.info(hyper_params)

        model_list = []
        val_scores = []
        results_val = {}
        results_val_list = []

        # Cross-Validation
        for fold in range(0, len(train_data_list)):
            train_data = train_data_list[fold]
            train_labels = train_labels_list[fold]
            
            val_data = val_data_list[fold]
            val_labels = val_labels_list[fold]

            test_data = test_data_list[fold]
            test_labels = test_labels_list[fold]

            data_unlabeled = data_unlabeled_list[fold]

            categorical_columns = categorical_columns_list[fold]

            hyper_params.class_weights = class_weights_list[fold]

            logging.info(f" ***** Validation Fold {fold} *****")

            logging.info(f"class weights (Fold {fold}) :" + str(hyper_params.class_weights))
            logging.info("Train data shape: " + str(train_data.shape))
            logging.info("Valid data shape: " + str(val_data.shape))
            logging.info("Test data shape: " + str(test_data.shape))

            model = XGBOOST(
                train_data=train_data, 
                train_labels=train_labels, 
                val_data=val_data, 
                val_labels=val_labels,
                test_data=test_data, 
                test_labels=test_labels, 
                data_unlabeled=data_unlabeled,
                categorical_columns=categorical_columns,
                hyper_params=hyper_params
            )

            model.train()

            results_val = model.valid()
            
            os.makedirs(
                Path( \
                    hyper_params.output_path, \
                    f"study_{str(trial.number)}",  \
                    f"Fold_{str(fold)}"
                ), exist_ok=True)

            model_file = open(Path( \
                            hyper_params.output_path, \
                            f"study_{str(trial.number)}",  \
                            f"Fold_{str(fold)}", \
                            "model_xgboost.pickle"), "wb")                                

            pickle.dump(model.model, model_file)
            
            output(
                output_path=Path(f"study_{str(trial.number)}", f"Fold_{str(fold)}"),
                hyper_params=hyper_params, 
                results=results_val
            )

            for key, value in results_val.items():
                logging.info(f"{key}: {value}")

            results_val_list.append(results_val)
            model_list.append(model)
        
        if hyper_params.is_regression:
            avg_r2_score = np.mean([score['val_r2_score'] for score in results_val_list])
            avg_rmse_score = np.mean([score['val_rmse_score'] for score in results_val_list])
            avg_mae_score = np.mean([score['val_mae_score'] for score in results_val_list])

            average_dict = {
                "avg_r2_score": avg_r2_score,
                "avg_rmse_score": avg_rmse_score,
                "avg_mae_score": avg_mae_score
            }

        else:
            raise

        for key, value in average_dict.items():
            logging.info(f"Avg {key}: " + str(value))

        output(
            output_path=Path(f"study_{str(trial.number)}", f"avg"),
            hyper_params=hyper_params, 
            results=average_dict
        )
        
        text = f"(Study {trial.number} in Test Fold {hyper_params.fold_number})\n\n"
        text += f"{hyper_params.output_path}\n\n"
        
        if hyper_params.is_regression:
            if hyper_params.metrics == "RMSE":
                text += f"Average RMSE Score: {str(avg_rmse_score)}\n\n"
            else:
                text += f"Average R2 Score: {str(avg_r2_score)}\n\n"
        else:
            raise
        
        text += "last fold:\n\n"

        for key, value in results_val.items():
            text += f"{key}: {value}\n\n"

        if hyper_params.is_regression:
            if hyper_params.metrics == "RMSE":
                return avg_rmse_score
            
            elif hyper_params.metrics == "R2SCORE":
                return avg_r2_score
        else:
            raise

    return objective

def testModel(train_data_list,
            train_labels_list,
            val_data_list,
            val_labels_list,
            test_data_list,
            test_labels_list,
            data_unlabeled_list,
            class_weights_list,
            categorical_columns_list,
            hyper_params,
            best_params):

    hyper_params.eta = best_params["eta"]        
    hyper_params.max_depth = int(best_params["max_depth"])
    hyper_params.min_child_weight = best_params["min_child_weight"]
    hyper_params.gamma = best_params["gamma"]
    hyper_params.colsample_bytree = best_params["colsample_bytree"]
    hyper_params.subsample = best_params["subsample"]

    logging.info(hyper_params)

    model_list = []
    results_test_list = []

    # Cross-Validation
    for fold in range(0, len(train_data_list)):
        
        train_data = train_data_list[fold]
        train_labels = train_labels_list[fold]
        
        val_data = val_data_list[fold]
        val_labels = val_labels_list[fold]

        test_data = test_data_list[fold]
        test_labels = test_labels_list[fold]

        data_unlabeled = data_unlabeled_list[fold]

        categorical_columns = categorical_columns_list[fold]

        hyper_params.fold_number_val = fold
        hyper_params.class_weights = class_weights_list[fold]

        logging.info(f" ***** Validation Fold {fold} *****")

        logging.info(f"class weights (Fold {fold}) :" + str(hyper_params.class_weights))
        logging.info("Train data shape: " + str(train_data.shape))
        logging.info("Valid data shape: " + str(val_data.shape))
        logging.info("Test data shape: " + str(test_data.shape))

        model = XGBOOST(
            train_data=train_data, 
            train_labels=train_labels, 
            val_data=val_data, 
            val_labels=val_labels,
            test_data=test_data, 
            test_labels=test_labels, 
            data_unlabeled=data_unlabeled,
            categorical_columns=categorical_columns,
            hyper_params=hyper_params
        )

        model.train()

        results_valid = model.valid()
        results_test = model.test()
        results_test_list.append(results_test)
        
        output(
            output_path=Path(f"test", f"Fold_{fold}"),
            hyper_params=hyper_params, 
            results=results_test
        )
        
        model_list.append(model)

    logging.info(results_test_list)

    if hyper_params.is_regression:
        avg_test_r2_score = np.mean([score['test_r2_score'] for score in results_test_list])
        avg_test_rmse_score = np.mean([score['test_rmse_score'] for score in results_test_list])
        avg_test_mae_score = np.mean([score['test_mae_score'] for score in results_test_list])

        average_dict = {
            "avg_test_r2_score": avg_test_r2_score,
            "avg_test_rmse_score": avg_test_rmse_score,
            "avg_test_mae_score": avg_test_mae_score
        }
        
    else:
        raise

    for key, value in average_dict.items():
        logging.info(f"Avg {key}: " + str(value))

    # WEB_HOOK_URL = ""  

    text = f"({hyper_params.config_name})\n"

    for key, value in average_dict.items():
        text += f"{key}: {value}\n\n"

    # requests.post(WEB_HOOK_URL, data=json.dumps({
    #     "text" : text
    # }))

@hydra.main(config_name="")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(cfg.pretty())
    hyper_params = cfg.trainer.hyper_params
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.trainer.gpu)

    seed = hyper_params.seed
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pl.seed_everything(seed)

    data_path = cfg.trainer.data_path
    fold_number = cfg.trainer.hyper_params.fold_number

    data_path = Path(data_path, str(fold_number))

    train_data_list = []
    train_labels_list = []

    val_data_list = []
    val_labels_list = []

    test_data_list = []
    test_labels_list = []

    data_unlabeled_list = []

    categorical_columns_list = []
    class_weights_list = []

    print(glob.glob(str(data_path) + "/[0-9]"))
    print(data_path)
    
    for fold_val_path in glob.glob(str(data_path) + "/[0-9]"):
        if hyper_params.is_used_raw_data:
            train_data = pd.read_csv(Path(fold_val_path, "train_data.csv") , index_col=0)
            val_data = pd.read_csv(Path(fold_val_path, "val_data.csv") , index_col=0)
            test_data = pd.read_csv(Path(fold_val_path, "test_data.csv") , index_col=0)
            data_unlabeled = pd.read_csv(Path(fold_val_path, "data_unlabeled.csv") , index_col=0)
            
        else:
            train_data = pd.read_csv(Path(fold_val_path, "train_data_std.csv") , index_col=0)
            val_data = pd.read_csv(Path(fold_val_path, "val_data_std.csv") , index_col=0)
            test_data = pd.read_csv(Path(fold_val_path, "test_data_std.csv") , index_col=0)
            data_unlabeled = pd.read_csv(Path(fold_val_path, "data_unlabeled_std.csv") , index_col=0)
        
        categorical_columns = pd.read_csv(Path(fold_val_path, "categorical_columns.csv"), index_col=0)

        train_labels = pd.read_csv(Path(fold_val_path, "train_labels.csv") , index_col=0)
        val_labels = pd.read_csv(Path(fold_val_path, "val_labels.csv") , index_col=0)
        test_labels = pd.read_csv(Path(fold_val_path, "test_labels.csv") , index_col=0)

        if hyper_params.is_regression:
            class_weights = []
            
        else:
            class_weights = class_weight.compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_labels), 
                    y=train_labels.values.ravel()
            ).tolist()


        train_data_list.append(train_data)
        train_labels_list.append(train_labels)

        val_data_list.append(val_data)
        val_labels_list.append(val_labels)

        test_data_list.append(test_data)
        test_labels_list.append(test_labels)

        data_unlabeled_list.append(data_unlabeled)

        categorical_columns_list.append(categorical_columns)
        class_weights_list.append(class_weights)

    pl.seed_everything(seed)

    current_date = datetime.now()
    current_day = current_date.strftime('%Y-%m-%d') 
    current_time = current_date.strftime('%H-%M-%S')
    
    current_dir = re.sub('outputs.+', "", os.getcwd()) + "/outputs"

    db_path = f"{current_dir}/optuna/db/"
    output_path = cfg.trainer.hyper_params.output_path
    output_path = re.sub('.+results/', "", output_path)

    db_path = Path(db_path, output_path)
    
    os.makedirs(db_path, exist_ok=True)

    path = Path(db_path, f"{str(fold_number)}.db")

    con = sqlite3.connect(path)
    con.close()

    sampler = TPESampler(seed=seed)

    optimize_direction = hyper_params.optimize_direction

    study = optuna.create_study(
        study_name=cfg.name,
        direction=optimize_direction,
        storage=f'sqlite:///{str(path)}',
        load_if_exists=True,
        sampler=sampler,
    )

    study.optimize(
        objective_with_data(
            train_data_list=train_data_list, 
            train_labels_list=train_labels_list, 
            val_data_list=val_data_list, 
            val_labels_list=val_labels_list,
            test_data_list=test_data_list, 
            test_labels_list=test_labels_list, 
            data_unlabeled_list=data_unlabeled_list,
            class_weights_list=class_weights_list,
            categorical_columns_list=categorical_columns_list,
            hyper_params=hyper_params), 
        n_trials=100)
    
    if cfg.trainer.debug:
        exit()
    
    print(study.best_params)
    print(study.best_value)

    hyper_params.config_name = cfg.trainer.config_name
    hyper_params.gpu = cfg.trainer.gpu

    # test
    testModel(train_data_list=train_data_list, 
            train_labels_list=train_labels_list, 
            val_data_list=val_data_list, 
            val_labels_list=val_labels_list,
            test_data_list=test_data_list, 
            test_labels_list=test_labels_list, 
            data_unlabeled_list=data_unlabeled_list,
            class_weights_list=class_weights_list,
            categorical_columns_list=categorical_columns_list,
            hyper_params=hyper_params,
            best_params=study.best_params)

if __name__ == '__main__':
    main()