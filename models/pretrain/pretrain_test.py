import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from configs.data_config.dataset_config import DataShapeConfig
from configs.data_config.project_config import ProjectConfig
from configs.train_config.pretrain_config import PretrainConfig
from data.train_data.dataset import CamelsDataset
from utils.model.test_full import test_full
from utils.model.tools import seed_torch

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    device = ProjectConfig.device
    num_workers = ProjectConfig.num_workers
    dataset_num_worker = ProjectConfig.dataset_num_worker
    use_board = ProjectConfig.use_board
    prefetch_factor = ProjectConfig.prefetch_factor

    seed = PretrainConfig.seed
    data_root = PretrainConfig.data_root
    saving_root = Path(PretrainConfig.saving_root)
    basin_root = PretrainConfig.basin_root
    used_model = PretrainConfig.used_model
    decode_mode = PretrainConfig.decode_mode
    n_epochs = PretrainConfig.n_epochs
    batch_size = PretrainConfig.batch_size
    learning_rate = PretrainConfig.learning_rate
    scheduler_paras = PretrainConfig.scheduler_paras
    best_model = PretrainConfig.model

    past_len = DataShapeConfig.past_len
    pred_len = DataShapeConfig.pred_len
    use_baseflow = DataShapeConfig.use_baseflow
    use_signatures = DataShapeConfig.use_signatures
    streamflow_size = DataShapeConfig.streamflow_size
    signatures_size = DataShapeConfig.signatures_size
    src_size = DataShapeConfig.src_size

    print("pid:", os.getpid())
    seed_torch(seed=seed)
    print(saving_root)
    # Model
    best_path = list(saving_root.glob(f"(max_sf_nse)*.pkl"))
    assert (len(best_path) == 1)
    best_path = best_path[0]
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    # Train mean and std to normalize data
    train_means = np.loadtxt(saving_root / "train_means.csv", dtype="float32")
    train_stds = np.loadtxt(saving_root / "train_stds.csv", dtype="float32")
    train_x_mean = train_means[:src_size]
    train_y_mean = train_means[src_size:]
    train_x_std = train_stds[:src_size]
    train_y_std = train_stds[src_size:]

    # Dataset
    basin_test = pd.read_excel(basin_root, sheet_name='test').iloc[:, 0].tolist()
    dataset_test = CamelsDataset(data_root, basin_test, past_len, pred_len, use_baseflow, use_signatures,
                                 device, dataset_num_worker, 'test',
                                 x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers,
                             prefetch_factor=prefetch_factor, shuffle=False)
    # Testing
    test_full(best_model, decode_mode, loader_test, device, saving_root,
              streamflow_size, signatures_size, use_baseflow, use_signatures)
