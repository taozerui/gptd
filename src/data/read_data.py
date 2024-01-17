import os
import pickle
import torch
import numpy as np
import pandas as pd
import joblib as jb
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader


def load_movie_file(file_path):
    data = {}
    for dt in ['train', 'test']:
        data[f'{dt}_ind'] = np.loadtxt(
            os.path.join(file_path, f'mv_{dt}_ind.txt')).astype(int)
        data[f'{dt}_val'] = np.loadtxt(
            os.path.join(file_path, f'mv_{dt}_y.txt')).astype(int)

    return data


def load_dblp_file(file_path):
    data = {}

    # train, validation
    dt = np.loadtxt(os.path.join(file_path, 'dblp-large-tensor.txt'))
    ind = dt[:, :3].astype(int)
    val = dt[:, -1].reshape(-1).astype(int)

    train_ind = ind
    train_val = val

    data['train_ind'] = train_ind
    data['train_val'] = train_val

    # test
    dt_test = loadmat(os.path.join(file_path, 'dblp.mat'))['data'][0][0][3]
    test_val = []
    test_ind = []
    for i in range(50):
        test_ind.append(dt_test[0][i][0][0][0] - 1)
        test_val.append(dt_test[0][i][0][0][1].reshape(-1))
    test_val = np.hstack(test_val).astype(np.int64)
    test_ind = np.vstack(test_ind).astype(np.int64)
    data['test_ind'] = test_ind
    data['test_val'] = test_val

    return data


def load_anime_file(file_path):
    data = {}

    dt = jb.load(os.path.join(file_path, 'anime_binary_1m.jb'))

    # train, validation
    train_ind = dt['train_X'].astype(int)
    train_val = dt['train_y'].astype(int)

    data['train_ind'] = train_ind
    data['train_val'] = train_val

    # test
    data['test_ind'] = dt['test_X'].astype(int)
    data['test_val'] = dt['test_y'].astype(int)

    return data


def load_kaggle_file(file_path):
    data = {}
    valid_train_ratio = 0.2

    dt = pd.read_csv(os.path.join(file_path, 'alldata.csv'))
    dt = dt.to_numpy()

    # train, validation
    ind = dt[:, :4].astype(int)
    val = dt[:, -1].astype(int)

    perm = np.random.permutation(ind.shape[0])
    ind = ind[perm]
    val = val[perm]
    valid_num = int(len(val) * valid_train_ratio)

    valid_ind = ind[:valid_num]
    valid_val = val[:valid_num]
    ind = ind[valid_num:]
    val = val[valid_num:]
    test_ind = ind[:valid_num]
    test_val = val[:valid_num]
    train_ind = ind[valid_num:]
    train_val = val[valid_num:]

    data['train_ind'] = train_ind
    data['train_val'] = train_val
    data['valid_ind'] = valid_ind
    data['valid_val'] = valid_val

    # test
    data['test_ind'] = test_ind
    data['test_val'] = test_val

    return data


def load_count_file(file_path, fold):
    with open(os.path.join(file_path, f'fold_{fold}.pkl'), 'rb') as f:
        dt = pickle.load(f)
    return dt


def load_enron_file(file_path, fold):
    dt = np.load(os.path.join(file_path, f'fold-{fold}.npz'))

    train_ind = dt['train'][:, :3].astype(int)
    train_val = dt['train'][:, -1].astype(int)
    test_ind = dt['test'][:, :3].astype(int)
    test_val = dt['test'][:, -1].astype(int)

    data = {
        'train_ind': train_ind, 'train_val': train_val,
        'test_ind': test_ind, 'test_val': test_val,
    }
    return data


def load_jhu_file(file_path, fold):
    dt = np.load(os.path.join(file_path, f'fold-{fold}.npz'))

    train_ind = dt['train'][:, :4].astype(int)
    train_val = dt['train'][:, -1].astype(float)
    test_ind = dt['test'][:, :4].astype(int)
    test_val = dt['test'][:, -1].astype(float)

    data = {
        'train_ind': train_ind, 'train_val': train_val,
        'test_ind': test_ind, 'test_val': test_val,
    }
    return data


def load_ems_file(file_path, fold):
    dt = np.load(os.path.join(file_path, f'fold-{fold}.npz'))

    train_ind = dt['train'][:, :2].astype(int)
    train_val = dt['train'][:, -1].astype(float)
    test_ind = dt['test'][:, :2].astype(int)
    test_val = dt['test'][:, -1].astype(float)

    data = {
        'train_ind': train_ind, 'train_val': train_val,
        'test_ind': test_ind, 'test_val': test_val,
    }
    return data


def get_discrete_data(file_path: str, batch_size=int(2**10), fold=None):

    dataset_name = file_path.split('/')
    if 'dblp' in dataset_name:
        raw_data = load_dblp_file(file_path)
    elif 'enron' in dataset_name or 'digg' in dataset_name or 'article' in dataset_name:
        assert fold is not None
        raw_data = load_enron_file(file_path, fold - 1)
    elif 'jhu' in dataset_name:
        assert fold is not None
        raw_data = load_jhu_file(file_path, fold - 1)
    elif 'ems' in dataset_name:
        assert fold is not None
        raw_data = load_ems_file(file_path, fold - 1)
    else:
        raise RuntimeError('Check dataset path and names!')

    train_ind = torch.tensor(raw_data['train_ind'], dtype=torch.int64)
    test_ind = torch.tensor(raw_data['test_ind'], dtype=torch.int64)

    train_val = torch.tensor(raw_data['train_val'], dtype=torch.int64)
    test_val = torch.tensor(raw_data['test_val'], dtype=torch.int64)

    train_dt = TensorDataset(train_ind, train_val)
    test_dt = TensorDataset(test_ind, test_val)

    train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dt, batch_size=1000, shuffle=False)

    data_loaders = {'train': train_loader, 'test': test_loader}

    return raw_data, data_loaders
