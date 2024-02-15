import os
from pathlib import Path
import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from FairInterventions.modules.CIFRank_module.generate_counterfactual_data import \
    get_counterfactual_data_real

project_dir = Path.cwd()


class CIFRank():
    def __init__(self, query_col, IV, MED, DV, control, out_path, pos_th=0, run=0):
        self.query_col = query_col
        self.IV = IV
        self.MED = MED
        self.DV = DV
        self.pos_th = pos_th
        self.control = control
        self.set_paths_specifics(out_path)
        self.set_paths_runs_specifics(run)

    def set_paths_specifics(self, out_path):
        self.causal_path_dir = os.path.join(out_path, "parameter_data")
        if not os.path.exists(self.causal_path_dir):
            os.makedirs(self.causal_path_dir)
        self.counter_path_dir = os.path.join(out_path, "counterfactual_data")
        if not os.path.exists(self.counter_path_dir):
            os.makedirs(self.counter_path_dir)

    def set_paths_runs_specifics(self, i):
        self.causal_path = os.path.join(self.causal_path_dir, str(i))
        self.counter_path = os.path.join(self.counter_path_dir, str(i))

    def generate_fair_data(self, data_train, data_test):
        IV_column = '_'.join(self.IV)

        data_train[IV_column] = data_train[self.IV[0]]
        for index in range(1, len(self.IV)):
            data_train[IV_column] = data_train[IV_column] + '_' + data_train[self.IV[index]]

        data_test[IV_column] = data_test[self.IV[0]]
        for index in range(1, len(self.IV)):
            data_test[IV_column] = data_test[IV_column] + '_' + data_test[self.IV[index]]

        self.IV = IV_column

        if not os.path.exists(self.causal_path):
            self.run_causal_model(data_train)

        if not os.path.exists(self.counter_path):
            get_counterfactual_data_real(data_train, self.causal_path, self.counter_path, 'count_train.csv', self.control,
                                        self.IV, self.DV, self.MED)
            get_counterfactual_data_real(data_test, self.causal_path, self.counter_path, 'count_test.csv',
                                        self.control,
                                        self.IV, self.DV, self.MED)

        counter_data_train = pd.read_csv(os.path.join(self.counter_path, 'count_train.csv'))
        counter_data_test = pd.read_csv(os.path.join(self.counter_path, 'count_test.csv'))

        return counter_data_train, counter_data_test

    def run_causal_model(self, data):
        data_pos = data[data[self.DV] > self.pos_th]
        try:
            pandas2ri.activate()
            r = robjects.r
            r_script = os.path.join(project_dir, "./FairInterventions/modules/CIFRank_module/R/estimate_causal_model.R")
            r.source(r_script, encoding="utf-8")
            r.estimate_causal_model(data_pos, self.IV, self.DV,
                                    self.MED, self.control, self.causal_path)
        except:
            if len(os.listdir(self.causal_path)) != 0:
                df = pd.DataFrame(columns=["mediators"])
                df["mediators"] = 'nan'
                df.to_csv(os.path.join(self.causal_path, 'identified_mediators.csv'))

            print("Save med results")
            self.save_med_results(self.control, data, self.causal_path)

    def save_med_results(self, control, temp, out_path):
        if os.path.exists(os.path.join(out_path, 'med_output.txt')):
            with open(os.path.join(out_path, 'med_output.txt'), 'r') as f:
                content = f.readlines()

            results_dict = dict()
            next_indirect = False
            for line in content:
                line = line.strip()
                if line.startswith('For the predictor'):
                    if len(results_dict.keys()) == 0:
                        pred = line.split(' ')[3]
                        df_med = pd.DataFrame(columns=['Metric', 'Estimate'])
                        results_dict[pred] = ''
                    else:
                        results_dict[pred] = df_med
                        pred = line.split(' ')[3]
                        df_med = pd.DataFrame(columns=['Metric', 'Estimate'])

                if line.startswith('The estimated total effect:'):
                    total_effect = float(line.split(' ')[4])
                    temp_df = pd.DataFrame([['Total Effect', total_effect]], columns=['Metric', 'Estimate'])
                    df_med = pd.concat([df_med, temp_df], ignore_index=True)

                if next_indirect:
                    splits = line.split(' ')
                    if splits[0] == '':
                        indirect_effect = float(line.split(' ')[1])
                    else:
                        indirect_effect = float(line.split(' ')[0])
                    temp_df = pd.DataFrame([['Indirect Effect', indirect_effect]], columns=['Metric', 'Estimate'])
                    df_med = pd.concat([df_med, temp_df], ignore_index=True)
                    next_indirect = False

                if line.startswith('y1.all'):
                    next_indirect = True

            results_dict[pred] = df_med

            pred_groups = [p.split('pred')[1] for p in results_dict.keys()]
            groups = temp[self.IV].unique()
            pred_gr = [g for g in groups if g not in pred_groups and g != control][0]
            index = 0
            print(results_dict)
            for key in results_dict.keys():
                index = index + 1
                df_med = results_dict[key]
                direct_effect = df_med[df_med['Metric'] == 'Total Effect']['Estimate'].values[0] - \
                                df_med[df_med['Metric'] == 'Indirect Effect']['Estimate'].values[0]
                temp_df = pd.DataFrame([['Direct Effect', direct_effect]], columns=['Metric', 'Estimate'])
                df_med = pd.concat([df_med, temp_df], ignore_index=True)

                if key == 'pred':
                    file_name = pred_gr + '_med.csv'
                elif 'pred.temp1$x' in key:
                    file_name = groups[index] + '_med.csv'
                else:
                    file_name = key.split('pred')[1] + '_med.csv'

                df_med.to_csv(os.path.join(out_path, file_name))
