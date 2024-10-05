import os
import random
import warnings

import numpy as np
import pandas as pd
import torch


# 设置torch的seed，实现可重现实验
def seed_torch(seed):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置python的哈希种子，相同的输入具有相同的哈希值
    np.random.seed(seed)
    torch.manual_seed(seed)  # 设置pytorch的随机种子，获得可复现的结果
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # 返回默认的卷积算法


# 计算给定mdl模型的总参数量
def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


def saving_obs_pred(obs, pred, start_date, end_date, past_len, pred_len, saving_root, date_index=None):
    if len(obs.shape) == 3 and obs.shape[-1] == 1:
        obs = obs.squeeze(-1)
    if len(pred.shape) == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if date_index is not None:
        pd_range = pd.date_range(start_date + pd.Timedelta(days=past_len), end_date - pd.Timedelta(days=pred_len - 1))
        date_index_seq = date_index[past_len:len(date_index) - pred_len + 1]
        if (len(date_index_seq) != len(pd_range)) or ((date_index_seq == pd_range).min is False):
            warnings.warn("The missing blocks are not contiguous and may cause some errors!")
        obs_pd = pd.DataFrame(obs, columns=[f"obs{i}" for i in range(obs.shape[1])], index=date_index_seq)
        pred_pd = pd.DataFrame(pred, columns=[f"pred{i}" for i in range(pred.shape[1])], index=date_index_seq)
    else:
        obs_pd = pd.DataFrame(obs, columns=[f"obs{i}" for i in range(obs.shape[1])])
        pred_pd = pd.DataFrame(pred, columns=[f"pred{i}" for i in range(pred.shape[1])])
    # print(obs_pd,pred_pd)

    saving_root.mkdir(parents=True, exist_ok=True)
    obs_pd.to_csv(saving_root / f"obs.csv", index=True, index_label="start_date")
    pred_pd.to_csv(saving_root / f"pred.csv", index=True, index_label="start_date")
