import os
import numpy as np
import pandas as pd

from FairInterventions.modules.LFR_module.LFR import LFR
from FairInterventions.utils.utils import writeToCSV


class LearningFairRepresentations():
    def __init__(self, query_col, score_col, sensitive_col, features_col,  k, Ax, Ay, Az, out_path, pos_th=0, run=0, model_occ=False):
        self.query_col = query_col
        self.score_col = score_col
        self.features_col = features_col,
        self.sensitive_col = sensitive_col
        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.pos_th = pos_th
        self.run = run
        self.model_occ = model_occ
        self.set_paths_specifics(out_path)
        self.set_paths_runs_specifics(run)

    def set_paths_specifics(self, out_path):
        self.fair_path_dir = os.path.join(out_path, "fair_data")
        self.model_logs_dir = os.path.join(out_path, "model_logs")

    def set_paths_runs_specifics(self, i):
        self.fair_path = os.path.join(self.fair_path_dir, str(i))
        self.model_logs_path = os.path.join(self.model_logs_dir, str(i))

    def generate_fair_data(self, data_train, data_test):
        if not os.path.exists(self.fair_path):
            os.makedirs(self.fair_path)

            if not os.path.exists(self.model_logs_path):
                os.makedirs(self.model_logs_path)

            data_train_fair = data_train.copy()
            data_test_fair = data_test.copy()

            features_cols = list(self.features_col)[0].copy()
            features_cols.append(self.score_col)

            model = LFR(k=self.k, Ax=self.Ax,
                        Ay=self.Ay,
                        Az=self.Az,
                        print_interval=100,
                        logs_path=os.path.join(self.model_logs_path, 'logs.txt'), verbose=1, seed=None)

            if not self.model_occ:
                self.fit_model(self, model, data_train_fair, features_cols)
                data_train_fair = model.transform(data_train, features_cols)
                data_test_fair = model.transform(data_test, features_cols)

            else:
                data_train_fair_list = []
                data_test_fair_list = []
                for qid in data_train_fair[self.query_col].unique():
                    if not os.path.exists(os.path.join(self.model_logs_path, str(qid))):
                        data_train_qid = data_train_fair[data_train_fair[self.query_col] == qid]
                        self.fit_model(model, data_train_qid, features_cols, qid)

                        data_test_qid = data_test_fair[data_test_fair[self.query_col] == qid]
                        data_train_transformed = model.transform(data_train_qid, features_cols)
                        data_test_transformed = model.transform(data_test_qid, features_cols)
                        data_train_fair_list.append(data_train_transformed)
                        data_test_fair_list.append(data_test_transformed)

                features_cols_fair = [col + '_fair' for col in features_cols]

                train_transformed = np.vstack(data_train_fair_list)
                test_transformed = np.vstack(data_test_fair_list)
                for index, col in enumerate(features_cols_fair):
                    data_train_fair.loc[:, col] = train_transformed[:, index]
                    data_test_fair.loc[:, col] = test_transformed[:, index]

            writeToCSV(os.path.join(self.fair_path, 'fair_train_data.csv'), data_train_fair)
            writeToCSV(os.path.join(self.fair_path, 'fair_test_data.csv'), data_test_fair)
        else:
            data_train_fair = pd.read_csv(os.path.join(self.fair_path, 'fair_train_data.csv'))
            data_test_fair = pd.read_csv(os.path.join(self.fair_path, 'fair_test_data.csv'))

        return data_train_fair, data_test_fair

    def fit_model(self, model, data_train, features_cols, qid=None):
        if qid:
            out_path = os.path.join(self.model_logs_path, str(qid))
        else:
            out_path = os.path.join(self.model_logs_path)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

            with open(os.path.join(out_path, 'logs.txt'), 'w') as f:
                f.write('Training Logs...')

            data_train[self.score_col] = data_train[self.score_col].apply(lambda x: 1 if x > self.pos_th else 0)
            print("Start model fit...")
            model.fit(data_train, features_cols, self.score_col, self.sensitive_col)

            with open(os.path.join(out_path, 'prototypes__model_parmas.npy'), 'wb') as f:
                np.save(f, model.prototypes)

            with open(os.path.join(out_path, 'weights__model_parmas.npy'), 'wb') as f:
                np.save(f, model.w)
        else:
            with open(os.path.join(self.model_logs_path, 'prototypes__model_parmas.npy'), 'rb') as f:
                model.prototypes = np.load(f)

            with open(os.path.join(self.model_logs_path, 'weights__model_parmas.npy'), 'rb') as f:
                model.w = np.load(f)