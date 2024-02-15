import pandas as pd
import os
from FairInterventions.utils.utils import writeToCSV

def get_counterfactual_data_real(cur_df, path_causal, output_path, file_name, control, IV, DV, MED):
    group_list = [x for x in cur_df[IV].unique() if x != control]

    orig_cols = list(cur_df.columns)

    mediators = pd.read_csv(os.path.join(path_causal, "identified_mediators.csv"))
    no_mediators = len(mediators) == 0 or str(mediators['mediators'].values[0]) == 'nan' or str(
        mediators['mediators'].values[0]) == 'x....cutx.'

    new_cols = []
    for med in MED:
        if med in mediators['mediators'].values:
            x_res = pd.read_csv(os.path.join(path_causal, med + "~" + IV + "-1.csv"))
            counter_g_base = x_res[x_res["Unnamed: 0"] == IV + control]["Estimate"].values[0]

            x_shifts = {control: 0}
            for gi in group_list:
                if not IV + gi in x_res["Unnamed: 0"].values:
                    x_shifts[gi] = 0
                else:
                    other_g_base = x_res[x_res["Unnamed: 0"] == IV + gi]["Estimate"].values[0]
                    x_shifts[gi] = counter_g_base - other_g_base

            feature_shifts = cur_df[IV].apply(lambda x: x_shifts[x])
            cur_df.loc[:, med + "_fair"] = cur_df[med] + feature_shifts
            new_cols.append(med + "_fair")

        else:
            # variables that are not mediators remain unchanged
            cur_df.loc[:, med + "_fair"] = cur_df[med]
            new_cols.append(med + "_fair")

    if no_mediators:
        # direct effect of the IV on the DV --> we keep the observed X as it is
        y_res = pd.read_csv(os.path.join(path_causal, DV + '~' + IV + "-1.csv"))
        counter_g_base = y_res[y_res["Unnamed: 0"] == IV + control]["Estimate"].values[0]
        y_shifts = {control: 0}
        for gi in group_list:
            if not IV + gi in y_res["Unnamed: 0"].values:
                y_shifts[gi] = 0
            else:
                y_shifts[gi] = counter_g_base - y_res[y_res["Unnamed: 0"] == IV + gi]["Estimate"].values[0]
    else:
        y_shifts = {control: 0}
        y_shifts_resolve = {control: 0}
        for gi in group_list:
            if not os.path.exists(os.path.join(path_causal, gi + "_med" + ".csv")):
                y_shifts[gi] = 0
                y_shifts_resolve[gi] = 0
            else:
                g_res = pd.read_csv(os.path.join(path_causal, gi + "_med" + ".csv"))
                y_shifts[gi] = -g_res[g_res['Metric'] == 'Total Effect']["Estimate"].values[0]
                y_shifts_resolve[gi] = -g_res[g_res['Metric'] == 'Direct Effect']["Estimate"].values[0]

    cur_df["Y_shift"] = cur_df[IV].apply(lambda x: y_shifts[x])
    cur_df[DV + "_fair"] = cur_df[DV] + cur_df["Y_shift"]
    new_cols.append(DV + "_fair")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_f = os.path.join(output_path, file_name)
    writeToCSV(output_f, cur_df)

    return cur_df