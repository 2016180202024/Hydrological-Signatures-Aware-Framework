import os

import torch
from torch.utils.tensorboard import SummaryWriter

from configs.data_config.project_config import ProjectConfig


# 保存最佳模型pkl
class BestModelLog:
    def __init__(self, init_model, saving_root, metric_name, high_better: bool,
                 log_all: bool = False):
        self.high_better = high_better
        self.saving_root = saving_root
        self.metric_name = metric_name
        worst = float("-inf") if high_better else float("inf")
        self.best_epoch = -1
        self.best_value = worst
        self.best_model_path = self.saving_root / f"({self.metric_name})_{self.best_epoch}_{self.best_value}.pkl"
        self.log_all = log_all
        if not self.log_all:
            if ProjectConfig.multi_gpu:
                torch.save(init_model.module.state_dict(), self.best_model_path)
            else:
                torch.save(init_model.state_dict(), self.best_model_path)

    def update(self, model, new_value, epoch):
        if self.log_all:
            self.best_value = new_value
            self.best_epoch = epoch
            self.best_model_path = self.saving_root / f"({self.metric_name})_{self.best_epoch}_{self.best_value}.pkl"
            torch.save(model.state_dict(), self.best_model_path)
            return
        if ((self.high_better is True) and (new_value > self.best_value)) or \
                ((self.high_better is False) and (new_value < self.best_value)):
            os.remove(self.best_model_path)
            self.best_value = new_value
            self.best_epoch = epoch
            self.best_model_path = self.saving_root / f"({self.metric_name})_{self.best_epoch}_{self.best_value}.pkl"
            torch.save(model.state_dict(), self.best_model_path)


# TensorBoard记录
class BoardWriter:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def write_board(self, msg, metric_value, epoch, every=2):
        if epoch % every == 0:
            self.writer.add_scalar(msg, metric_value, global_step=epoch)
            self.writer.close()
