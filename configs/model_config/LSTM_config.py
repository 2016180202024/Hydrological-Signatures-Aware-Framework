from configs.data_config.dataset_config import DataShapeConfig


class LSTMConfig:
    model_name = "LSTM"
    decode_mode = None
    seq_len_e = 0
    output_len_e = 0
    input_size_e = 0
    hidden_size_e = 128
    output_len_d = 0
    output_size = 0
    hidden_size_d = 128
    dropout_rate = 0.2

    def __init__(self, src_len=DataShapeConfig.src_len, src_size=DataShapeConfig.src_size,
                 past_len=DataShapeConfig.past_len, pred_len=DataShapeConfig.pred_len,
                 tgt_size=DataShapeConfig.tgt_size):
        self.seq_len_e = src_len
        self.output_len_e = pred_len
        self.input_size_e = src_size
        self.output_len_d = pred_len
        self.output_size = tgt_size

    model_info = (f"{model_name}_"
                  f"[hs1_{hidden_size_e},hs3_{hidden_size_d},dr_{dropout_rate}]")
