import numpy as np
import torch
from utils.model.metrics import calc_nse, calc_nrmse


def eval_model(model, data_loader, decode_mode, device,
               streamflow_size, signatures_size):
    """
    Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param data_loader: A PyTorch DataLoader, providing the data.
    :param decode_mode: autoregressive or non-autoregressive
    :param device: device for data and models
    :param streamflow_size: features size of timeseries streamflow
    :param signatures_size: features size of static signatures

    :return: loss, streamflow_nse, signatures_mse
    分别为：总损失、streamflow损失、signatures损失
    """
    # set model to eval mode (important for dropout)
    model.eval()
    # 输出预测结果
    obs_all, preds_all = predict_by_dataloader_and_model(model, data_loader, decode_mode, device)
    # rescale
    obs_all = data_loader.dataset.local_rescale(obs_all.numpy(), variable='y')
    preds_all = data_loader.dataset.local_rescale(preds_all.numpy(), variable='y')
    obs_all[obs_all < 0] = 0
    preds_all[preds_all < 0] = 0
    # 计算streamflow nse
    obs_streamflow, pred_streamflow = (
        obs_all[:, :, :streamflow_size], preds_all[:, :, :streamflow_size]
    )
    sf_nse_all = np.nanmean(calc_nse(obs_streamflow, pred_streamflow), axis=0)
    sf_nse = sf_nse_all[0]
    # 计算baseflow nse
    bf_nse = [0, 0]
    if streamflow_size > 1:
        bf_nse = sf_nse_all[1:streamflow_size].tolist()
    # 计算signatures mse
    sg_mse = 0
    if signatures_size != 0:
        obs_signatures, pred_signatures = (
            obs_all[:, :, -signatures_size:], preds_all[:, :, -signatures_size:]
        )
        sg_mse = np.nanmean(calc_nrmse(obs_signatures, pred_signatures))
    return sf_nse, bf_nse, sg_mse


def predict_by_dataloader_and_model(model, data_loader, decode_mode, device):
    """
    Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param data_loader: A PyTorch DataLoader, providing the data.
    :param decode_mode: autoregressive or non-autoregressive
    :param device: device for data and modelsd

    :return: Three torch Tensors, containing:
    observations y, predictions y, all samples y_std
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    with torch.no_grad():
        for x_seq, y_seq_past, y_seq_future, _ in data_loader:
            # print(f"[{index}]: predicting data %.4f" % (index / len(data_loader)))
            x_seq, y_seq_past, y_seq_future = x_seq.to(device), y_seq_past.to(device), y_seq_future.to(device)
            batch_size = y_seq_past.shape[0]
            tgt_len = y_seq_past.shape[1] + y_seq_future.shape[1]
            tgt_size = y_seq_future.shape[2]
            pred_len = y_seq_future.shape[1]

            enc_inputs = x_seq
            dec_inputs = torch.zeros((batch_size, tgt_len, tgt_size)).to(device)
            dec_inputs[:, :-pred_len, :] = y_seq_past
            # get model predictions
            if decode_mode == "NAR":
                y_hat = model(enc_inputs, dec_inputs)
                y_hat = y_hat[:, -pred_len:, :]
            elif decode_mode == "AR":
                for i in range(tgt_len - pred_len, tgt_len):
                    decoder_predict = model(enc_inputs, dec_inputs)
                    dec_inputs[:, i, :] = decoder_predict[:, i - 1, :]
                y_hat = dec_inputs[:, -pred_len:, :]
            else:  # Model is not Transformer
                y_hat = model(x_seq, y_seq_past)

            obs.append(y_seq_future.to("cpu"))
            preds.append(y_hat.to("cpu"))

    obs_all = torch.cat(obs)
    preds_all = torch.cat(preds)

    return obs_all, preds_all
