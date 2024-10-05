# Define the shape of data
class DataShapeConfig:
    # 输入输出时间序列长度
    # past_len [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    # 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
    past_len = 140  # TODO Length of formerly known runoff sequence
    pred_len = 30  # TODO Length of runoff sequence to be predicted
    tgt_len = past_len + pred_len
    src_len = past_len + pred_len
    # 输入特征长度
    dynamic_input_size = 16
    static_input_size = 37
    src_size = dynamic_input_size + static_input_size  # input attributes size
    # 输出特征长度
    tgt_size = 1  # Number of target attributes
    use_baseflow = True  # TODO: whether to use baseflow
    use_signatures = True  # TODO: whether to use static signature
    baseflow_output_size = 2
    static_output_size = 14
    streamflow_size = 1
    signatures_size = 0
    if use_baseflow is True:
        tgt_size += baseflow_output_size
        streamflow_size += baseflow_output_size
    if use_signatures is True:
        tgt_size += static_output_size
        signatures_size += static_output_size

    data_shape_info = (f"[{past_len}-{pred_len},{src_size}-{tgt_size}]"
                       f"_bf{use_baseflow}_sn{use_signatures}")
