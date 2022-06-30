import os
import pickle
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
import pytorch_lightning as pl

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

class XGBOOST():
    def __init__(self, 
                train_data, 
                train_labels,
                val_data, 
                val_labels, 
                test_data, 
                test_labels, 
                data_unlabeled,
                categorical_columns,
                hyper_params):

        self.hyper_params = hyper_params
        self.class_weights = hyper_params.class_weights
        self.is_regression = hyper_params.is_regression

        self.evals_result = {}

        pl.seed_everything(self.hyper_params.seed)

        if self.is_regression:
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                seed=self.hyper_params.seed,
                n_estimators=self.hyper_params.num_round,
                max_depth=self.hyper_params.max_depth,
                learning_rate=self.hyper_params.eta,
                min_child_weight=self.hyper_params.min_child_weight,
                gamma=self.hyper_params.gamma,
                subsample=self.hyper_params.subsample,
                colsample_bytree=self.hyper_params.colsample_bytree,
                random_state=self.hyper_params.seed,
                tree_method='gpu_hist', 
                gpu_id=hyper_params.gpu
            )
        
        self.X_train = train_data
        self.y_train = train_labels

        self.X_val = val_data
        self.y_val = val_labels

        self.X_test = test_data
        self.y_test = test_labels

        self.data_unlabeled = data_unlabeled

        self.categorical_columns = categorical_columns

        if self.is_regression:
            self.val_sample_weights = []
            self.sample_weights = []
            
        else:
            self.val_sample_weights = [[self.class_weights[int(element)]] for element in val_labels.values.ravel()]
            self.sample_weights = [[self.class_weights[int(element)]] for element in train_labels.values.ravel()]

        self.data_train_and_unlabeled = self.X_train.copy()
        self.data_train_and_unlabeled = self.data_train_and_unlabeled.append(self.data_unlabeled)

        self.data_all = self.data_train_and_unlabeled.append(self.X_val)
        self.data_all = self.data_all.append(self.X_test)

        if not self.hyper_params.is_used_one_hot_data:
            for column in self.X_train.columns.values:
                if column in self.categorical_columns:
                    self.X_train = self.X_train.astype({column: int})
                    self.X_test = self.X_test.astype({column: int})
                    self.X_val = self.X_val.astype({column: int})
                    self.data_train_and_unlabeled = self.data_train_and_unlabeled.astype({column: int})
                    
                    self.X_train[column] = self.X_train[column] - 1
                    self.X_test[column] = self.X_test[column] - 1
                    self.X_val[column] = self.X_val[column] - 1
                    self.data_train_and_unlabeled[column] = self.data_train_and_unlabeled[column] - 1
                    self.data_all[column] = self.data_all[column] - 1
                    
                    self.le = LabelEncoder()
                    self.le.fit(self.data_all[column].unique())

                    self.X_train[column] = self.le.transform(self.X_train[column].values)
                    self.X_val[column] = self.le.transform(self.X_val[column].values)
                    self.X_test[column] = self.le.transform(self.X_test[column].values)
                    self.data_train_and_unlabeled[column] = self.le.transform(self.data_train_and_unlabeled[column].values)

    def train(self):
        logging.info(f"{self.__class__.__name__}")

        if self.is_regression:
            eval_metric = "rmse"
            
            self.model = self.model.fit(
                X=self.X_train.values, 
                y=self.y_train.values.flatten(),
                eval_metric=eval_metric,
                eval_set=[[self.X_val.values, self.y_val.values.flatten()]],
                early_stopping_rounds=20,
                verbose=True
            )

        self.evals_result = self.model.evals_result()
        
        print("RESULT:", self.evals_result)
        
        # print(self.model.best_iteration)
        # print(self.model.best_score)

    def valid(self):
        y_pred = self.model.predict(self.X_val.values)
        y_true = self.y_val
        logging.info(f"*** Valid Result *** ")

        print(y_pred)
        print(y_true.values)

        if self.is_regression:
            val_r2_score = r2_score(y_true, y_pred)
            val_rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
            val_mae_score = mean_absolute_error(y_true, y_pred)

            logging.info(f"val_r2_score: {val_r2_score}")
            logging.info(f"val_rmse_score: {val_rmse_score}")
            logging.info(f"val_mae_score: {val_mae_score}")

            return {
                "val_r2_score": val_r2_score,
                "val_rmse_score": val_rmse_score,
                "val_mae_score": val_mae_score
            }

    def test(self):
        y_pred = self.model.predict(self.X_test.values)
        y_true = self.y_test

        os.makedirs(Path(self.hyper_params.output_path, f"test", f"Fold_{self.hyper_params.fold_number_val}"), exist_ok=True)

        model_file = open(Path(self.hyper_params.output_path, f"test", f"Fold_{self.hyper_params.fold_number_val}", \
                          "model_xgboost.pickle"), "wb")

        pickle.dump(self.model, model_file)

        if self.is_regression: 
            test_r2_score = r2_score(y_true, y_pred)
            test_rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
            test_mae_score = mean_absolute_error(y_true, y_pred)

            logging.info(f"test_r2_score: {test_r2_score}")
            logging.info(f"test_rmse_score: {test_rmse_score}")
            logging.info(f"test_mae_score: {test_mae_score}")

            return {
                "test_r2_score": test_r2_score,
                "test_rmse_score":test_rmse_score,
                "test_mae_score": test_mae_score
            }
    
    def predict(self, data):
        if not isinstance(data, (np.ndarray, np.generic)):
            data = data.values

        y_pred = self.model.predict(data)
        y_pred = np.round(y_pred)
        
        return y_pred