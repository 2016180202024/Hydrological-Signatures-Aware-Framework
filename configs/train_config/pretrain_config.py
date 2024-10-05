import os.path
import importlib

import torch
from torch import nn

from configs.data_config.path_config import PathConfig
from configs.data_config.dataset_config import DataShapeConfig
from configs.data_config.project_config import ProjectConfig
from utils.model import streamflow_loss
from utils.model.lr_strategies import SchedulerFactory


class PretrainConfig:
    stage = 'train'  # TODO train test

    # Random seed config
    seed = 1234
    device = ProjectConfig.device

    # training config
    scale_factor = 1  # TODO: the bath_size bigger, the learning_rate larger.
    n_epochs = 200  # TODO: origin 200
    batch_size = 4096 // scale_factor  # TODO
    learning_rate = 0.001 / scale_factor  # TODO: lr=0.0002
    learning_rate_weights = 0.001  # TODO

    # use model config
    used_model = "LSTMMSVS2S"  # TODO Transformer LSTMMSVS2S
    # 先修改model_config
    model_congfigs = importlib.import_module(f"configs.model_config.{used_model}_config")
    ModelConfig = getattr(model_congfigs, f"{used_model}Config")
    model_config = ModelConfig()
    # 后新建model
    models = importlib.import_module("models")
    Model = getattr(models, used_model)
    model = Model(model_config)
    if ProjectConfig.multi_gpu:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)

    # loss function config
    if DataShapeConfig.use_baseflow:
        weights = [1, 1]
        if DataShapeConfig.use_signatures:
            weights = [1, 1, 1]
        loss_func = streamflow_loss.StreamflowLoss(model, weights).to(device)
    else:
        loss_func = streamflow_loss.StreamflowLoss().to(device)

    # Optimizer, Scheduler config
    scheduler_paras = {"scheduler_type": "warm_up", "last_epoch": -1,
                       "warm_up_epochs": n_epochs * 0.25, "decay_rate": 0.99}
    if DataShapeConfig.use_baseflow or DataShapeConfig.use_signatures:
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate},
                                      {'params': loss_func.weights, 'lr': learning_rate_weights}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = SchedulerFactory.get_scheduler(optimizer, **scheduler_paras)
    # scheduler_paras = {"scheduler_type": "none", "last_epoch": -1,}
    # scheduler_paras = {"scheduler_type": "exp_decay", "last_epoch": -1,
    #                    "decay_epoch": n_epochs * 0.5, "decay_rate": 0.99}
    # scheduler_paras = {"scheduler_type": "cos_anneal", "last_epoch": -1,
    #                    "cos_anneal_t_max": 32}

    # save message config
    learning_config_info = f"n{n_epochs}_bs{batch_size}_lr{learning_rate}_lrw{learning_rate_weights}"
    decode_mode = ModelConfig.decode_mode
    saving_message = f"{ModelConfig.model_info}@{DataShapeConfig.data_shape_info}" \
                     f"@{learning_config_info}@seed{seed}"

    # input and output path config
    days = DataShapeConfig.past_len + DataShapeConfig.pred_len
    data_root = os.path.join(PathConfig.train_path, str(days))
    basin_root = os.path.join(PathConfig.model_conf_path, f'basin_list_{stage}.xlsx')
    saving_root = os.path.join(PathConfig.model_path, str(days), saving_message)
    if stage == 'test':
        saving_root = os.path.join(PathConfig.model_path, stage, saving_message)
