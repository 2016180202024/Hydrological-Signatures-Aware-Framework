import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from configs.data_config.dataset_config import DataShapeConfig
from configs.data_config.project_config import ProjectConfig
from configs.train_config.pretrain_config import PretrainConfig
from data.train_data.dataset import CamelsDataset
from utils.model.tools import seed_torch
from utils.model.train_full import train_full


if __name__ == '__main__':
    device = ProjectConfig.device
    num_workers = ProjectConfig.num_workers
    dataset_num_worker = ProjectConfig.dataset_num_worker
    use_board = ProjectConfig.use_board
    prefetch_factor = ProjectConfig.prefetch_factor
    pin_memory = ProjectConfig.pin_memory
    use_train_eval = ProjectConfig.use_train_eval

    seed = PretrainConfig.seed
    data_root = PretrainConfig.data_root
    saving_root = Path(PretrainConfig.saving_root)
    basin_root = PretrainConfig.basin_root
    decode_mode = PretrainConfig.decode_mode
    n_epochs = PretrainConfig.n_epochs
    batch_size = PretrainConfig.batch_size

    loss_func = PretrainConfig.loss_func
    model = PretrainConfig.model
    optimizer = PretrainConfig.optimizer
    scheduler = PretrainConfig.scheduler

    past_len = DataShapeConfig.past_len
    pred_len = DataShapeConfig.pred_len
    use_baseflow = DataShapeConfig.use_baseflow
    use_signatures = DataShapeConfig.use_signatures
    streamflow_size = DataShapeConfig.streamflow_size
    signatures_size = DataShapeConfig.signatures_size

    print("pid:", os.getpid())
    seed_torch(seed=seed)
    saving_root.mkdir(exist_ok=True, parents=True)
    print(saving_root)
    torch.autograd.set_detect_anomaly(True)

    basin_train = pd.read_excel(basin_root, sheet_name='train').iloc[:, 0].tolist()
    basin_val = pd.read_excel(basin_root, sheet_name='val').iloc[:, 0].tolist()

    dataset_train = CamelsDataset(data_root, basin_train, past_len, pred_len,
                                  use_baseflow, use_signatures, device, dataset_num_worker)
    dataset_val = CamelsDataset(data_root, basin_val, past_len, pred_len,
                                use_baseflow, use_signatures, device, dataset_num_worker,
                                'val', dataset_train.get_means()[0], dataset_train.get_means()[1],
                                dataset_train.get_stds()[0], dataset_train.get_stds()[1])
    # We use the feature means/stds of the training data for normalization in val and test stage
    train_x_mean, train_y_mean = dataset_train.get_means()
    train_x_std, train_y_std = dataset_train.get_stds()
    # Saving training mean and training std
    train_means = np.concatenate((train_x_mean, train_y_mean), axis=0)
    train_stds = np.concatenate((train_x_std, train_y_std), axis=0)
    np.savetxt(saving_root / "train_means.csv", train_means)
    np.savetxt(saving_root / "train_stds.csv", train_stds)
    with open(saving_root / "y_stds_dict.pickle", "wb") as f:
        pickle.dump(dataset_train.y_stds_dict, f)
    # dataset_train, dataset_val = random_split(
    #     dataset_train,
    #     [round(0.8 * len(dataset_train)), round(0.2 * len(dataset_train))],
    #     generator=torch.Generator().manual_seed(seed)
    # )

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory)

    # model.load_state_dict(torch.load(
    #     r'D:\experiment\model\60\Transformer_NAR_[64-4-4-256-0.1]@[30-30,53-1]_bfFalse_snFalse@n200_bs1024_lr0.001@seed1234\(max_sf_nse)_113_0.5689130425453186.pkl', map_location=device))

    # Training and Validation
    train_full(model, decode_mode, loader_train, loader_val, optimizer,
               scheduler, loss_func, n_epochs, device, saving_root, use_board, use_train_eval,
               use_baseflow, use_signatures, streamflow_size, signatures_size)
