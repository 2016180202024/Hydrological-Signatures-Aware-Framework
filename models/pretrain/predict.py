import os.path
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import datetime as dt

from pandas import compat

from configs.data_config.path_config import PathConfig
from configs.data_config.project_config import ProjectConfig
from utils.model.buid_model import build_model_with_name_datashape, normalization, read_data_shape_by_model_name
from utils.process_data.date_transform import date_to_doy
from configs.data_config.extract_config import ExtractConfig

streamflow_columns = ExtractConfig.streamflow_columns
forcing_columns = ExtractConfig.forcing_columns
device = ProjectConfig.device


def predict_by_xy_seq_data(model, model_name, train_mean, train_stds,
                           x_seq, y_seq, past_len, pred_len, src_size, pred_size):
    # normalization x_seq & y_seq
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x_mean, y_mean = train_mean[:src_size], train_mean[src_size:]
        x_std, y_std = train_stds[:src_size], train_stds[src_size:]
        x_seq = torch.tensor(normalization(x_seq, x_mean, x_std), dtype=torch.float32).to(device)
        if model_name == 'LSTM' or model_name == 'TransformerS':
            y_hat = model(x_seq, None)
        else:
            y_seq = torch.tensor(normalization(y_seq, y_mean, y_std), dtype=torch.float32).to(device)
            if 'LSTM' in model_name:
                y_hat = model(x_seq, y_seq[:, :past_len, :])
            else:
                y_hat = model(x_seq, y_seq)[:, -pred_len:, :]
    y_hat = y_hat.cpu().numpy()
    y_hat = y_hat * y_std + y_mean
    y_sign = None
    if pred_size > 3:
        y_sign = y_hat[:, :, 3:]
        y_hat = y_hat[:, :, :3]
    return y_hat, y_sign


def get_xy_by_data(static_data, modis_data, sign_data, ts_data,
                   past_len, pred_len, src_size, pred_size):
    all_len = past_len + pred_len
    x_seq = np.empty(shape=(all_len, src_size), dtype=np.float32)
    y_seq = np.empty(shape=(all_len, pred_size), dtype=np.float32)
    if pred_size <= 3:
        y_seq = np.array(ts_data.loc[:, streamflow_columns[:pred_size]])
    else:
        y_seq[:, :3] = np.array(ts_data.loc[:, streamflow_columns[:pred_size]])
        y_seq[:, 3:] = np.repeat(sign_data.reshape(1, -1), all_len, axis=0)
    first_index = ts_data.index[0]
    for days in range(all_len):
        date = ts_data.loc[days + first_index, 'date'].date()
        doy = int(date_to_doy(date))
        if date >= dt.date(2000, 1, 1):
            date_str = str(date.year) + '%02d' % date.month
            lai = modis_data['lai_' + date_str]
            lai_maxdiff = modis_data['lai_' + str(date.year) + '_maxdiff']
            evi = modis_data['evi_' + date_str]
            evi_maxdiff = modis_data['evi_' + str(date.year) + '_maxdiff']
        else:
            month = '%02d' % date.month
            lai = modis_data['lai_' + month]
            lai_maxdiff = modis_data['lai_maxdiff']
            evi = modis_data['evi_' + month]
            evi_maxdiff = modis_data['evi_maxdiff']
        x_seq[days, :5] = [doy, lai, lai_maxdiff, evi, evi_maxdiff]
    x_seq[:, 5:5 + len(forcing_columns)] = np.array(ts_data.loc[:, forcing_columns])
    x_seq[:, 5 + len(forcing_columns):] = np.repeat(static_data.reshape(1, -1), all_len, axis=0)
    return x_seq, y_seq


def predict(camels_dict, model_path,
            input_path=PathConfig.origin_path):
    input_path = Path(input_path)
    dir_path = os.path.dirname(model_path)
    dir_name = os.path.basename(dir_path)
    print('--------------------------------------')
    print(f'[{dir_name}] is predicting!')
    output_dir_path = Path(dir_path) / 'predict'
    # 加载模型的输入数据类型
    data_shape = read_data_shape_by_model_name(dir_name)
    past_len, pred_len = data_shape['past_len'], data_shape['pred_len']
    all_len = past_len + pred_len
    src_size, pred_size = data_shape['src_size'], data_shape['pred_size']
    pred_size_ts = 1 if pred_size == 1 else 3
    output_columns = streamflow_columns[:1] if pred_size == 1 else streamflow_columns
    # 加载模型
    model_name = str.split(dir_name, '_')[0]
    datashape = {'src_len': all_len, 'src_size': src_size, 'past_len': past_len,
                 'pred_len': pred_len, 'tgt_size': pred_size}
    model = build_model_with_name_datashape(model_name, datashape)
    state_dict = {}
    for k, v in torch.load(model_path).items():
        state_dict[k[7:]] = v
    model.load_state_dict(state_dict)
    # 加载训练数据的x_mean x_std
    train_mean = np.loadtxt(os.path.join(dir_path, 'train_means.csv'), dtype="float32")
    train_std = np.loadtxt(os.path.join(dir_path, 'train_stds.csv'), dtype="float32")
    # 计算全部len
    all_length = 0
    for camels in camels_dict:
        all_length += len(camels_dict[camels])
    temp_length = 0
    zero_len_gauge_list = []
    # 输入：camels_static.csv (static & modis & signatures)
    for camels in camels_dict:
        camels_output_dir_path = output_dir_path / camels
        camels_output_dir_path.mkdir(parents=True, exist_ok=True)
        camels_path = input_path / camels
        static_path = camels_path / 'static'
        static_data = pd.read_csv(str(static_path / (camels + '_static_features.csv'))).set_index('gauge_id')
        modis_data = pd.read_csv(str(static_path / (camels + '_modis_features.csv'))).set_index('gauge_id')
        sign_data = pd.read_csv(str(static_path / (camels + '_streamflow_signatures.csv'))).set_index('gauge_id')
        gauge_list = camels_dict[camels]
        # 输入：gauge_id.csv (timeseries)
        # 输出：gauge_id.csv (true & predict)
        for gauge_id in gauge_list:
            temp_length += 1
            output_file_path = camels_output_dir_path / (gauge_id + '_pred.csv')
            if output_file_path.exists():
                print(f'[{temp_length}/{all_length}] [{gauge_id}] is exists!')
                continue
            # read gauge_id.csv timeseries data
            ts_path = camels_path / 'timeseries' / (gauge_id + '.csv')
            ts_data = pd.read_csv(str(ts_path), index_col='date')
            if '-' in str(ts_data.index[0]):
                ts_data.index = pd.to_datetime(ts_data.index, format='%Y-%m-%d')
            else:
                ts_data.index = pd.to_datetime(ts_data.index, format='%Y/%m/%d')
            start_date = ts_data.index[0]
            end_date = ts_data.index[-1]
            print(f'[{temp_length}/{all_length}] [{gauge_id}]: ({start_date},{end_date}）')
            # prepare output data
            df_date = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
            ts_data = pd.merge(df_date, ts_data, how='left',
                               left_on=df_date.index, right_on=ts_data.index)
            ts_data = ts_data.rename(columns={'key_0': 'date'})
            static_gauge_data = np.array(static_data.loc[gauge_id, :])
            modis_gauge_data = modis_data.loc[gauge_id, :].to_dict()
            sign_gauge_data = np.array(sign_data.loc[gauge_id, :])
            start = 0
            end = past_len
            length = ts_data.shape[0]
            index_arr = []
            x_seq_all = []
            y_seq_all = []
            while end <= length - pred_len:
                # 1. 处理nan值，获取数据的开始结束的index
                streamflow = ts_data.loc[start:end, output_columns]  # 在选定时间范围中的时间序列数据，如60天
                # 如果在past_len中有nan值，那么预测第一个nan值及其后面的,并且start变为第一个notnan或end
                if np.isnan(np.array(streamflow)).any():
                    nan_index = streamflow.index[np.where(np.isnan(streamflow))[0]]
                    first_nan_index = nan_index.values[0]
                    last_nan_index = nan_index.values[-1]
                    # 输入序列为[nanindex-pastlen, nanindex+pred_len]
                    pred_start = first_nan_index - past_len
                    pred_end = first_nan_index + pred_len
                    start = last_nan_index + 1
                    end = start + past_len
                    if pred_start <= 0:
                        continue
                # 如果没有nan值,正常预测
                else:
                    pred_start = start
                    pred_end = end + pred_len
                    start = start + pred_len
                    end = end + pred_len
                # 2. 根据数据获取x_seq和y_seq，并拼接到输入数据中
                temp_ts_data = ts_data.iloc[pred_start:pred_end, :]
                x_seq, y_seq = get_xy_by_data(
                    static_gauge_data, modis_gauge_data, sign_gauge_data, temp_ts_data,
                    past_len, pred_len, src_size, pred_size)
                if np.isnan(x_seq).any() or np.isnan(y_seq[:past_len, :]).any():
                    continue
                index_arr.append(pred_end)
                x_seq_all.append(x_seq)
                y_seq_all.append(y_seq)
            # 最后处理一次，以防有遗留
            temp_ts_data = ts_data.iloc[length - all_len:length, :]
            x_seq, y_seq = get_xy_by_data(
                static_gauge_data, modis_gauge_data, sign_gauge_data, temp_ts_data,
                past_len, pred_len, src_size, pred_size)
            if not (np.isnan(x_seq).any() or np.isnan(y_seq[:past_len, :]).any()):
                index_arr.append(length)
                x_seq_all.append(x_seq)
                y_seq_all.append(y_seq)
            # 3. 将x_seq和y_seq输入模型，并得到输出y_hat, y_sign
            if len(x_seq_all) <= 1:
                zero_len_gauge_list.append(gauge_id)
                continue
            y_hat, _ = predict_by_xy_seq_data(model, model_name, train_mean, train_std,
                                              np.array(x_seq_all), np.array(y_seq_all),
                                              past_len, pred_len, src_size, pred_size)
            # 4. 将y_hat, y_sign根据index_array拼接到输出中
            y_ts_pred = np.full(shape=(length, pred_size_ts), fill_value=np.nan)
            for index in range(len(index_arr)):
                end = index_arr[index]
                start = end - pred_len
                y_ts_pred[start:end, :] = y_hat[index, :, :]
            y_ts_pred[y_ts_pred < 0] = 0
            # 5. 输出文件
            pred_output_columns = [x + '_pred' for x in output_columns]
            output_columns1 = ['date'] + output_columns
            ts_data = ts_data.loc[:, output_columns1]
            ts_data.loc[:, pred_output_columns] = y_ts_pred
            ts_data.to_csv(str(output_file_path), index=False)
    print(zero_len_gauge_list)
    np.savetxt(str(output_dir_path / 'unused_gauge.csv'), np.array(zero_len_gauge_list),
               delimiter=',', fmt='%s')


def tibet_predict():
    camels_dict = {'tibet': ['tibet_JZ', 'tibet_LS', 'tibet_LZ', 'tibet_NGS',
                             'tibet_NX', 'tibet_PD', 'tibet_RKZ', 'tibet_TJ']}
    r'D:\experiment\model\tibet\30\LSTMMSVS2S_[hs1_128,hs2_128,hs3_128,dr_0.2]@[29-1,53-1]_bfFalse_snFalse@n100_bs512_lr0.001@seed1234/(max_sf_nse)_35_0.9857933297381809.pkl'
    r'D:\experiment\model\tibet\60\LSTMMSVS2S_[hs1_128,hs2_128,hs3_128,dr_0.2]@[57-3,53-1]_bfFalse_snFalse@n100_bs512_lr0.001@seed1234/(max_sf_nse)_32_0.968344400514383.pkl'
    r'D:\experiment\model\tibet\90\LSTMMSVS2S_[hs1_128,hs2_128,hs3_128,dr_0.2]@[83-7,53-1]_bfFalse_snFalse@n200_bs512_lr0.001@seed1234/(max_sf_nse)_58_0.9490402146549712.pkl'

    r'D:\experiment\model\tibet\30\LSTM_[hs1_128,hs3_128,dr_0.2]@[29-1,53-1]_bfFalse_snFalse@n200_bs512_lr0.001@seed1234/(max_sf_nse)_10_0.7017939049716573.pkl'
    r'D:\experiment\model\tibet\60\LSTM_[hs1_128,hs3_128,dr_0.2]@[57-3,53-1]_bfFalse_snFalse@n200_bs512_lr0.001@seed1234/(max_sf_nse)_33_0.7102608987497084.pkl'
    r'D:\experiment\model\tibet\90\LSTM_[hs1_128,hs3_128,dr_0.2]@[83-7,53-1]_bfFalse_snFalse@n200_bs512_lr0.001@seed1234/(max_sf_nse)_49_0.7095513868294098.pkl'

    model_path = \
        r'D:\experiment\model\tibet\90\LSTM_[hs1_128,hs3_128,dr_0.2]@[83-7,53-1]_bfFalse_snFalse@n200_bs512_lr0.001@seed1234/(max_sf_nse)_49_0.7095513868294098.pkl'
    predict(camels_dict, model_path)


def load_used_basin_dict():
    basin_list_path = Path(PathConfig.model_path) / 'config' / 'basin_list.csv'
    camels_dict = pd.read_csv(basin_list_path).to_dict(orient='list')
    unused_basin_list_path = Path(PathConfig.model_path) / 'config' / 'unused_basin_list.csv'
    unused_basin_list = pd.read_csv(unused_basin_list_path)['gauge_id'].tolist()
    for camels in camels_dict:
        camels_dict[camels] = [x for x in camels_dict[camels] if not (x is np.nan or x in unused_basin_list)]
    return camels_dict


def camels_predict():
    camels_dict = load_used_basin_dict()
    dir_path = Path(PathConfig.model_path)
    # '30', '60', '90', '120'
    for days in ['30']:
        days_dir_path = dir_path / days
        for model in ['LSTMMSVS2S', 'Transformer']:
            for model_dir_path in days_dir_path.glob(f'{model}_*'):
                model_path = list(model_dir_path.glob(f"(max_sf_nse)*.pkl"))
                assert (len(model_path) == 1)
                predict(camels_dict, model_path[0])


if __name__ == '__main__':
    camels_predict()
