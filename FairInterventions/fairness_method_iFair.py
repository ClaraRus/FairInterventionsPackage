import os
import pandas as pd
from flatbuffers.builder import np

from FairInterventions.utils.utils import writeToCSV


class iFairRanking():
    def __init__(self, query_col, score_col, sensitive_col, features_col,  k, Ax, Az, out_path, pos_th=0, run=0, max_iter=100, nb_restarts=3, batch_size=1000, model_occ=False):
        self.query_col = query_col
        self.score_col = score_col
        self.sensitive_col = sensitive_col
        self.features_col = features_col
        self.k = k
        self.Ax = Ax
        self.Az = Az
        self.run = run
        self.max_iter = max_iter
        self.nb_restarts = nb_restarts
        self.batch_size = batch_size
        self.pos_th = pos_th
        self.model_occ = model_occ
        self.set_paths_specifics(out_path)
        self.set_paths_runs_specifics(run)

    def set_paths_specifics(self, out_path):
        self.ifair_path_dir = os.path.join(out_path, "iFair_data")

    def set_paths_runs_specifics(self, i):
        self.ifair_path = os.path.join(self.ifair_path_dir, str(i))

    def generate_fair_data(self, data_train, data_test):
        from src.modules.iFair_module.iFair import iFair

        if not os.path.exists(self.ifair_path):
            os.makedirs(self.ifair_path)

        if not os.path.exists(os.path.join(self.ifair_path, 'iFair_train_data.csv')):
            distances_path = os.path.join(self.ifair_path, 'distances')
            model = iFair(distances_path, k=self.k, A_x=self.Ax,
                          A_z=self.Az, max_iter=self.max_iter,
                          nb_restarts=self.nb_restarts)

            features_cols = self.features_col.copy()
            features_cols.append(self.score_col)

            nonsensitive_column_indices = list(range(0, data_train[features_cols].shape[1]))

            features_cols.append(self.sensitive_col + '_coded')

            data_train_fair = data_train.copy()
            data_test_fair = data_test.copy()

            codes, uniques = pd.factorize(data_train_fair[self.sensitive_col])
            data_train_fair[self.sensitive_col + '_coded'] = codes

            codes, uniques = pd.factorize(data_test_fair[self.sensitive_col])
            data_test_fair[self.sensitive_col + '_coded'] = codes

            run = self.ifair_path.split('/')[-1]

            if not self.model_occ:
                self.fit_model(self, model, data_train, features_cols, nonsensitive_column_indices, run)
                data_train_fair = model.transform(data_train_fair[features_cols].to_numpy())
                data_test_fair = model.transform(data_test_fair[features_cols].to_numpy())

            else:
                data_train_fair_list = []
                data_test_fair_list = []
                qids = data_train_fair[self.query_col].unique()
                for qid in qids:
                    df_qid_train = data_train_fair[data_train_fair[self.query_col] == qid]
                    df_qid_test = data_test_fair[data_test_fair[self.query_col] == qid]

                    if not os.path.exists(os.path.join(self.ifair_path, str(qid) + '__model_parmas.npy')):
                        self.fit_model(model, df_qid_train, features_cols, nonsensitive_column_indices, run, qid)

                    data_train_transformed = model.transform(df_qid_train[features_cols].to_numpy())
                    data_test_transformed = model.transform(df_qid_test[features_cols].to_numpy())
                    data_train_fair_list.append(data_train_transformed)
                    data_test_fair_list.append(data_test_transformed)

                features_cols_fair = [col + '_fair' for col in features_cols]

                train_transformed = np.vstack(data_train_fair_list)
                test_transformed = np.vstack(data_test_fair_list)
                for index, col in enumerate(features_cols_fair):
                    data_train_fair.loc[:, col] = train_transformed[:, index]
                    data_test_fair.loc[:, col] = test_transformed[:, index]

            writeToCSV(os.path.join(self.ifair_path, 'iFair_train_data.csv'), data_train_fair)
            writeToCSV(os.path.join(self.ifair_path, 'iFair_test_data.csv'), data_test_fair)
        else:
            data_train_fair = pd.read_csv(os.path.join(self.ifair_path, 'iFair_train_data.csv'))
            data_test_fair = pd.read_csv(os.path.join(self.ifair_path, 'iFair_test_data.csv'))
        return data_train_fair, data_test_fair

    def fit_model(self, model, data_train, features_cols, nonsensitive_column_indices, run, qid=None):
        if qid is not None:
            out_path = os.path.join(self.ifair_path, str(qid) + '__model_parmas.npy')
        else:
            out_path = os.path.join(self.ifair_path, 'model_parmas.npy')

        if os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                model.opt_params = np.load(f)
        else:
            temp = data_train[data_train[self.score_col] > self.pos_th]
            model.fit(temp[features_cols].to_numpy(), run, qid,
                      batch_size=self.batch_size,
                      nonsensitive_column_indices=nonsensitive_column_indices)
            with open(out_path, 'wb') as f:
                np.save(f, model.opt_params)