import os
import pickle
import shutil
import json
import time
import torch
import argparse
import numpy as np
import yaml
from omegaconf import OmegaConf
from pathlib import Path

from src.utils import Trainer
from src.data import get_discrete_data
from src.model import RBF, SolveGPTFPolyaGamma


def main():
    parser = argparse.ArgumentParser(description='Tensor Completion')
    parser.add_argument('--dataset', type=str, default='dblp',
                        help='name of the dataset',
                        choices=['dblp', 'enron', 'digg', 'jhu', 'article', 'ems'])
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1024,
                        help='random seed')
    parser.add_argument('--debug', action='store_true', help='Debug')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # read config
    with open(f'./config/{args.dataset}.yaml') as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)
    rank = conf.model.rank
    induce = conf.model.num_inducing

    # writer
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    for i in range(args.repeat):
        if not args.debug:
            log_path = os.path.join(
                './log', args.dataset, f'rank_{rank}', f'm_{induce}', f'run-{now}', f'run-{i}'
            )
            Path(log_path).mkdir(parents=True, exist_ok=True)
            for sub_folder in ['data', 'config', 'checkpoints', 'code']:
                Path(os.path.join(log_path, sub_folder)).mkdir(
                    parents=True, exist_ok=True)
        else:
            log_path = None
        main_run_func(args, conf, log_path, i+1)


def main_run_func(args, conf, log_path, fold):
    # read data
    file_name = os.path.join('./dataset/processed', args.dataset)
    data, data_loader = get_discrete_data(
        file_name, batch_size=conf.train.batch_size, fold=fold
    )

    # saving
    if not args.debug:
        # save code
        save_code_path = os.path.join(log_path, 'code')
        shutil.copytree('./src', os.path.join(save_code_path, 'src'))
        shutil.copy('./main.py', os.path.join(save_code_path, 'main.py'))
        # save args
        save_arg_path = os.path.join(log_path, 'config', 'args.txt')
        with open(save_arg_path, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # kernel
    kernel = RBF(conf.kernel.band_width)
    # model
    mconf = conf.model
    zeta = mconf.zeta if 'zeta' in mconf.keys() else None
    model = SolveGPTFPolyaGamma(
        tensor_shape=mconf.tensor_shape,
        data_num=data_loader['train'].dataset.tensors[0].shape[0],
        data_type=mconf.data_type,
        kernel=kernel,
        rank=mconf.rank,
        num_inducing=mconf.num_inducing,
        prior_precision=mconf.prior_precision,
        zeta=zeta,
        init_lr=mconf.init_lr,
        adapt_lr=mconf.adapt_lr,
        n_mc=mconf.n_mc
    )
    if torch.cuda.is_available():
        model = model.cuda()

    # trainer
    train_conf = conf.train
    optimizer = torch.optim.Adam(model.parameters(), lr=train_conf.lr)
    trainer = Trainer(
        model=model, conf=conf, optimizer=optimizer, log_path=log_path, print_eval=True
    )
    trainer.train(
        data_loader['train'],
        test_loader=data_loader['test']
    )

    result_path = os.path.join(log_path, 'test_result.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(trainer.log_test_metric, f)


if __name__ == "__main__":
    main()
