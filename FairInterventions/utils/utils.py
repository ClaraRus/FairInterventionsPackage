import pickle
import os, pathlib, json
from sklearn.metrics import pairwise


def writeToTXT(file_name_with_path, _df):
    # try:
    #     _df.to_csv(file_name_with_path, header=False, index=False, sep=' ')
    # except FileNotFoundError:
    directory = os.path.dirname(file_name_with_path)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    print("Make folder ", directory)
    _df.to_csv(file_name_with_path, header=False, index=False, sep=' ')


def writeToCSV(file_name_with_path, _df):
    try:
        _df.to_csv(file_name_with_path, index=False)
    except FileNotFoundError:

        directory = os.path.dirname(file_name_with_path)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Make folder ", directory)
        _df.to_csv(file_name_with_path, index=False)


def writeToJson(file_name_with_path, _data):
    directory = os.path.dirname(file_name_with_path)
    if not os.path.exists(directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Make folder ", directory)
    with open(file_name_with_path, 'w') as fp:
        json.dump(_data, fp, indent=2)


def readFromJson(file_name_with_path, return_key=None):
    with open(file_name_with_path) as json_data:
        d = json.load(json_data)
    if return_key:
        return d[return_key]
    else:
        return d


def check_nan(df, cols_train):
    for col in cols_train:
        if df[col].isnull().values.any():
            return True
    return False


def compute_euclidean_distances(X, euclidean_dist_dir, batch, nonsensitive_column_indices):
    if not nonsensitive_column_indices:
        nonsensitive_column_indices = list(range(0, X.shape[1] - 1))

    if not os.path.exists(euclidean_dist_dir):
        os.makedirs(euclidean_dist_dir)
        if batch is not None:
            for i in range(0, len(X), batch):
                D_X_F = pairwise.euclidean_distances(X[i:i + batch, nonsensitive_column_indices],
                                                     X[:, nonsensitive_column_indices])
                with open(os.path.join(euclidean_dist_dir,
                                       'euclidean_distance_' + str(i) + '_' + str(i + batch)) + '.pkl',
                          'wb') as f:
                    pickle.dump(D_X_F, f)
        else:
            D_X_F = pairwise.euclidean_distances(X[:, nonsensitive_column_indices],
                                                 X[:, nonsensitive_column_indices])
            with open(os.path.join(euclidean_dist_dir, 'euclidean_distance.pkl'), 'wb') as f:
                pickle.dump(D_X_F, f)
        return D_X_F