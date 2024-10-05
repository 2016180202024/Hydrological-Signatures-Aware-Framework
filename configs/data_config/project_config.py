import os
import torch


# Project root, computing resources
class ProjectConfig:
    multi_gpu = True
    # multi_gpu = (torch.cuda.device_count() > 1)  # TODO: multi gpu to run
    if multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    num_workers = 4  # TODO: number of threads for loading data
    dataset_num_worker = 4
    prefetch_factor = 4  # TODO
    pin_memory = False  # TODO
    use_board = True  # TODO: production environment is False
    use_train_eval = False
