import os
from pathlib import Path

import numpy as np
import pandas as pd

from configs.data_config.path_config import PathConfig
from models.pretrain.predict import load_used_basin_dict
from utils.model.metrics import CalcEvalIndex


def evaluate(camels_dict, model_path):
    dir_path = os.path.dirname(model_path)
    print('--------------------------------------')
    print(f'[{os.path.basename(dir_path)}] is predicting!')
    output_file_path = Path(dir_path) / 'predict' / 'evaluation.csv'
    # 如果有已处理的gauge，读取然后跳过这些gauge
    exists_gauge = []
    eval_data = pd.DataFrame(columns=['gauge_id', 'nse', 'rmse', 'kge', 'bias', 'tpe5', 'R']).set_index('gauge_id')
    if output_file_path.exists():
        eval_data = pd.read_csv(output_file_path).set_index('gauge_id')
        exists_gauge = eval_data.index.tolist()
    # 计算所有len
    all_length = 0
    for camels in camels_dict:
        all_length += len(camels_dict[camels])
    temp_length = 0
    # 遍历每一个camels和gauge
    try:
        for camels in camels_dict:
            for gauge_id in camels_dict[camels]:
                temp_length += 1
                if gauge_id in exists_gauge:
                    print(f'[{temp_length}/{all_length}] [{gauge_id}] is exists!')
                    continue
                print(f'[{temp_length}/{all_length}] [{gauge_id}]')
                data_file_path = Path(dir_path) / 'predict' / camels / f'{gauge_id}_pred.csv'
                guage_data = pd.read_csv(data_file_path)
                obs_streamflow, pred_streamflow = guage_data['streamflow'], guage_data['streamflow_pred']
                cal_stream = CalcEvalIndex(obs=obs_streamflow, sim=pred_streamflow, calc_median=False)
                rmse_mean, _ = cal_stream.calc_rmse()
                nse_mean, _ = cal_stream.calc_nse()
                kge_mean, _ = cal_stream.calc_kge()
                bias_mean, _ = cal_stream.calc_bias()
                tpe5_mean, _ = cal_stream.calc_tpe_1D(5)
                R_mean, _ = cal_stream.calc_R()
                temp_eval_data = np.array([nse_mean, rmse_mean, kge_mean, bias_mean, tpe5_mean, R_mean])
                eval_data.loc[gauge_id] = temp_eval_data
    except Exception as e:
        print(e)
    finally:
        eval_data.to_csv(output_file_path)


def camels_evaluate():
    camels_dict = load_used_basin_dict()
    dir_path = Path(PathConfig.model_path)
    # '30', '60', '90', '120'
    for days in ['30']:
        days_dir_path = dir_path / days
        for model in ['LSTMMSVS2S', 'Transformer']:
            for model_dir_path in days_dir_path.glob(f'{model}_*'):
                model_path = list(model_dir_path.glob(f"(max_sf_nse)*.pkl"))
                assert (len(model_path) == 1)
                evaluate(camels_dict, model_path[0])


if __name__ == '__main__':
    camels_evaluate()
