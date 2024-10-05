# All metrics needs obs and sim shape: (batch_size, pred_len, streamflow_size)

import numpy as np
import torch


# 先计算成二维矩阵，行坐标为天，纵坐标为数据
# 再按行坐标平均mean计算特征统计
class CalcEvalIndex:
    def __init__(self, obs, sim, calc_median=True):
        self.obs, self.sim = self.drop_nan(obs, sim)
        self.mean_obs = np.mean(self.obs, axis=0)
        self.mean_sim = np.mean(self.sim, axis=0)
        self.std_obs = np.std(self.obs, axis=0)
        self.std_sim = np.std(self.sim, axis=0)
        self.calc_median = calc_median

    def calc_nse(self):
        denominator = np.mean((self.obs - self.mean_obs) ** 2, axis=0)
        diff = (self.sim - self.obs) ** 2
        mean = np.mean(diff, axis=0)
        mean = 1 - mean / denominator
        median = None
        if self.calc_median:
            median = np.median(diff, axis=0)
            median = 1 - median / denominator
        return mean, median

    def calc_R(self):
        diff_obs = self.obs - self.mean_obs
        diff_sim = self.sim - self.mean_sim
        denominator = (np.sqrt(np.mean(diff_sim ** 2, axis=0)) *
                       np.sqrt(np.mean(diff_obs ** 2, axis=0)))
        mean = np.mean(diff_obs * diff_sim, axis=0) / denominator
        median = None
        if self.calc_median:
            median = np.median(diff_obs * diff_sim, axis=0) / denominator
        return mean, median

    def calc_kge(self):
        beta = self.mean_sim / self.mean_obs
        alpha = self.std_sim / self.std_obs
        denominator = self.std_obs * self.std_sim
        diff = (self.obs - self.mean_obs) * (self.sim - self.mean_sim)
        mean = np.mean(diff, axis=0) / denominator
        mean = 1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (mean - 1) ** 2)
        median = None
        if self.calc_median:
            median = np.median(diff, axis=0) / denominator
            median = 1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (median - 1) ** 2)
        return mean, median

    # alpha是从高到低的n%的top
    def calc_tpe(self, alpha):
        shape = self.obs.shape
        mean_all = np.empty((shape[1], shape[2]), dtype=np.float32)
        median_all = np.empty((shape[1], shape[2]), dtype=np.float32)
        top = int(shape[0] * alpha / 100.0)
        for index in range(shape[2]):
            obs_temp = self.obs[:, :, index]
            sim_temp = self.sim[:, :, index]
            sort_index = np.argsort(obs_temp, axis=0)
            obs_sort = np.take_along_axis(obs_temp, sort_index, axis=0)
            sim_sort = np.take_along_axis(sim_temp, sort_index, axis=0)
            obs_t = obs_sort[-top:, :]
            sim_t = sim_sort[-top:, :]
            denominator = np.mean(obs_t, axis=0)
            bias = np.abs(sim_t - obs_t)
            mean = np.mean(bias, axis=0) / denominator
            mean_all[:, index] = mean
            if self.calc_median:
                median = np.median(bias, axis=0) / denominator
                median_all[:, index] = median
        return mean_all, median_all

    # alpha是从高到低的n%的top
    def calc_tpe_1D(self, alpha):
        shape = self.obs.shape
        top = int(shape[0] * alpha / 100.0)
        sort_index = np.argsort(self.obs, axis=0)
        obs_sort = np.take_along_axis(self.obs, sort_index, axis=0)
        sim_sort = np.take_along_axis(self.sim, sort_index, axis=0)
        obs_t = obs_sort[-top:]
        sim_t = sim_sort[-top:]
        denominator = np.mean(obs_t, axis=0)
        bias = np.abs(sim_t - obs_t)
        mean = np.mean(bias, axis=0) / denominator
        median = None
        if self.calc_median:
            median = np.median(bias, axis=0) / denominator
        return mean, median

    def calc_bias(self):
        bias = self.sim - self.obs
        mean = np.mean(bias, axis=0)
        mean = mean / self.mean_obs
        median = None
        if self.calc_median:
            median = np.median(bias, axis=0)
            median = median / self.mean_obs
        return mean, median

    def calc_rmse(self):
        mse = (self.obs - self.sim) ** 2
        mean = np.sqrt(np.mean(mse, axis=0))
        median = None
        if self.calc_median:
            median = np.sqrt(np.median(mse, axis=0))
        return mean, median

    def calc_nrmse(self):
        mse = (self.obs - self.sim) ** 2
        diff = np.max(self.obs, axis=0) - np.min(self.obs, axis=0)
        mean = np.sqrt(np.mean(mse, axis=0)) / diff
        median = None
        if self.calc_median:
            median = np.sqrt(np.median(mse, axis=0)) / diff
        return mean, median

    def drop_nan(self, array1: np.array, array2: np.array):
        index_array = []
        for index in range(array1.shape[0]):
            if np.isnan(array1[index]).any() or np.isnan(array2[index]).any():
                index_array.append(index)
        return (np.delete(array1, index_array, axis=0),
                np.delete(array2, index_array, axis=0))


def calc_nse(obs, sim):
    mean_obs = np.nanmean(obs, axis=0)
    denominator = np.nanmean((obs - mean_obs) ** 2, axis=0)
    diff = (sim - obs) ** 2
    nse = 1 - np.mean(diff, axis=0) / denominator
    return nse


def calc_nrmse(obs, sim):
    obs = np.nanmean(obs, axis=1)
    sim = np.nanmean(sim, axis=1)
    rmse = np.sqrt(np.nanmean((obs - sim) ** 2, axis=0))
    max = np.max(obs, axis=0)
    min = np.min(obs, axis=0)
    nrmse = np.divide(rmse, max - min, out=np.zeros_like(rmse), where=(max - min != 0))
    rmse[np.isinf(rmse)] = 0
    return nrmse


if __name__ == '__main__':
    # obs1 = np.load(r'C:\Users\admini\Desktop\obs.npy')
    # sim1 = np.load(r'C:\Users\admini\Desktop\sim.npy')
    # cal = CalcEvalIndex(obs=obs1, sim=sim1)
    # rmse_mean, rmse_median = cal.calc_rmse()
    # nse_mean, nse_median = cal.calc_nse()
    # kge_mean, kge_median = cal.calc_kge()
    # bias_mean, bias_median = cal.calc_bias()
    # tpe5_mean, tpe5_median = cal.calc_tpe(5)
    rmse = np.array([1, 2, 3, 4, np.inf, np.nan])
    rmse[np.isinf(rmse)] = 0
    rmse[np.isnan(rmse)] = 0
    rmse[rmse is np.nan] = 0
    print(rmse)
