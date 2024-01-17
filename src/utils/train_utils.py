import time
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
import sklearn.metrics as metrics


class Trainer:
    """docstring for Trainer."""
    def __init__(
        self,
        model,
        conf,
        optimizer,
        log_path=None,
        print_eval=True,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.conf = conf
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=conf.train.lr
        ) if optimizer is None else optimizer
        miles = [int(i * conf.train.epoch) for i in conf.train.mile_stones]
        self.miles = miles
        self.opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer, milestones=miles, gamma=0.3
        )
        self.log_path = log_path

        self.print_eval = print_eval

        self.save_model_interval = conf.train.save_model_int
        self.eval_interval = conf.train.eval_int

        self.current_epoch = 0
        self.current_iter = 0
        self.start_time = None

        if self.log_path is not None:
            self.writer = SummaryWriter(log_dir=log_path)
        else:
            self.writer = None

        self.log_test_metric = dict(
            auc=[], acc=[], rmse=[], mae=[], ll=[],
            fscore=[], mape=[]
        )

        self.save_config()

    def save_config(self):
        if self.log_path is not None:
            config_path = os.path.join(self.log_path, 'config', 'config.yaml')
            OmegaConf.save(config=self.conf, f=config_path)
        else:
            pass

    def save_model(self, best=False):
        if self.save_model_interval <= 0 or self.log_path is None:
            return 0

        epoch = self.current_epoch
        is_save = epoch % self.save_model_interval == 0 or \
            epoch == self.conf.train.epoch - 1
        if not is_save:
            return 0

        if best:
            model_path = os.path.join(
                self.log_path, 'checkpoints', 'best.pt'
            )
        else:
            model_path = os.path.join(
                self.log_path, 'checkpoints', f'epoch-{epoch}.pt'
            )
        torch.save(self.model.state_dict(), Path(model_path))

    def train(self, train_loader, valid_loader=None, test_loader=None):
        bar = tqdm(range(self.conf.train.epoch), desc='[Epoch 0]')
        self.start_time = time.time()
        for epoch in bar:
            bar.set_description(f'[Epoch {epoch}]')
            tic = time.time()
            self.train_epoch(train_loader)
            toc = time.time()
            print(f'Time is {toc - tic}s.')
            self.opt_scheduler.step()
            bar.set_postfix({'Loss': self.current_loss})
            self.save_model()
            if epoch in self.miles:
                if hasattr(self.model, 'lr') and not self.conf.model.adapt_lr:
                    self.model.lr *= 0.3

            is_eval = epoch % self.eval_interval == 0 or \
                epoch == self.conf.train.epoch - 1

            if is_eval:
                if valid_loader is not None:
                    self.eval_epoch(valid_loader, 'Valid')
                if test_loader is not None:
                    self.eval_epoch(test_loader, 'Test')

            self.current_epoch += 1

    def train_epoch(self, data_loader):
        model = self.model
        writer = self.writer
        epoch = self.current_epoch

        model.train()

        loss_log = []
        bar = tqdm(data_loader, desc='[Iter 0]', leave=False)
        for batch_idx, (inputs, x_val) in enumerate(bar):
            if torch.cuda.is_available():
                inputs, x_val = inputs.cuda(), x_val.cuda()

            elbo = model(inputs, x_val)
            loss = - elbo
            if batch_idx % 10:
                bar.set_postfix({'Loss': loss.item()})
                bar.set_description(f'[Iter {batch_idx}]')

            self.optimizer.zero_grad()
            loss.backward()
            if hasattr(self.conf.train, 'grad_clip'):
                if self.current_epoch < self.miles[0] and self.conf.train.grad_clip > 0.:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.conf.train.grad_clip,
                        norm_type=self.conf.train.grad_clip_norm
                    )
            self.optimizer.step()

            if hasattr(self.conf.train, 'log_grad_norm'):
                if self.conf.train.log_grad_norm and writer is not None:
                    gnorm_2 = gnorm_inf = 0.
                    for param in model.parameters():
                        if param.grad is not None:
                            gnorm_2 += param.grad.data.norm(2).item() ** 2
                            gnorm_inf = np.maximum(param.grad.data.norm(torch.inf).item(), gnorm_inf)
                    gnorm_2 = np.sqrt(gnorm_2)
                    writer.add_scalar('Grad/L2-norm', gnorm_2, self.current_iter)
                    writer.add_scalar('Grad/Inf-norm', gnorm_inf, self.current_iter)

            loss_log.append(loss.item())
            self.current_iter += 1

        loss_log = np.mean(loss_log)
        self.current_loss = loss_log

        if writer is not None:
            writer.add_scalar('Loss/Train', loss_log, epoch)
            writer.add_scalar(
                'Status/LR', self.opt_scheduler.get_last_lr()[0], epoch)

    @torch.no_grad()
    def eval_epoch(self, data_loader, phase):
        model = self.model
        writer = self.writer
        epoch = self.current_epoch

        model.eval()
        x_hat_tot = []
        x_val_tot = []
        logp_tot = []
        for _, (inputs, x_val) in enumerate(data_loader):
            if torch.cuda.is_available():
                inputs, x_val = inputs.cuda(), x_val.cuda()

            x_hat, logp = model(inputs, x_val, predict=True)
            x_hat_tot.append(x_hat.view(-1))
            x_val_tot.append(x_val.view(-1))
            logp_tot.append(logp.view(-1))

        x_hat_tot = torch.cat(x_hat_tot).detach().cpu().numpy()
        x_val_tot = torch.cat(x_val_tot).cpu().numpy()
        logp_tot = torch.cat(logp_tot).cpu().numpy()
        ll = np.mean(logp_tot)

        if self.conf.model.data_type == 'binary':
            roauc = metrics.roc_auc_score(y_true=x_val_tot, y_score=x_hat_tot)

            thresh_opt = 0.5
            x_hat_idx = np.zeros(x_hat_tot.shape[0])
            x_hat_idx[x_hat_tot < thresh_opt] = 0
            x_hat_idx[x_hat_tot >= thresh_opt] = 1

            fscore = metrics.f1_score(y_true=x_val_tot, y_pred=x_hat_idx)
            acc = metrics.accuracy_score(y_true=x_val_tot, y_pred=x_hat_idx)
            if writer is not None:
                writer.add_scalar(f'{phase}/AUC', roauc, epoch)
                writer.add_scalar(f'{phase}/ACC', acc, epoch)
                writer.add_scalar(f'{phase}/F-Score', fscore, epoch)
                writer.add_scalar(f'{phase}/LL', ll, epoch)
            if self.print_eval:
                print(
                    f'Epoch {epoch} - {phase}: AUC is {roauc:.3f} | ACC is {acc:.3f} | LL is {ll:.3f}.')
            self.log_test_metric['auc'].append(roauc)
            self.log_test_metric['acc'].append(acc)
            self.log_test_metric['ll'].append(ll)
            self.log_test_metric['fscore'].append(fscore)
        elif self.conf.model.data_type == 'count':
            rmse = np.sqrt(np.mean(np.square(x_hat_tot - x_val_tot))) / np.sqrt(np.mean(x_val_tot ** 2))
            mae = np.mean(np.abs(x_hat_tot - x_val_tot)) / np.mean(np.abs(x_val_tot))
            mape = np.mean(np.abs(x_hat_tot - x_val_tot) / np.abs(x_val_tot + 1.))
            if writer is not None:
                writer.add_scalar(f'{phase}/RMSE', rmse, epoch)
                writer.add_scalar(f'{phase}/MAE', mae, epoch)
                writer.add_scalar(f'{phase}/MAPE', mape, epoch)
                writer.add_scalar(f'{phase}/LL', ll, epoch)
            if self.print_eval:
                print(
                    f'Epoch {epoch} - {phase}: RMSE is {rmse:.3f} | MAE is {mae:.3f} | MAPE is {mape:.3f} | LL is {ll:.3f}.')
            self.log_test_metric['rmse'].append(rmse)
            self.log_test_metric['mae'].append(mae)
            self.log_test_metric['mape'].append(mape)
            self.log_test_metric['ll'].append(ll)
        else:
            pass
