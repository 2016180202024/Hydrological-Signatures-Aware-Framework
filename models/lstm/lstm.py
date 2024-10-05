import torch.nn as nn
from configs.model_config.LSTM_config import LSTMConfig


class EncoderRNN1(nn.Module):
    def __init__(self, seq_len_e1, output_len_e1, input_size_e1, hidden_size_e1):
        super().__init__()
        self.seq_len_e1 = seq_len_e1
        self.output_len_e1 = output_len_e1
        self.input_size_e1 = input_size_e1
        self.hidden_size_e1 = hidden_size_e1

        self.lstm = nn.LSTM(input_size=self.input_size_e1, hidden_size=self.hidden_size_e1, num_layers=1, bias=True,
                            dropout=0, batch_first=True, bidirectional=False)

    def forward(self, inputs):
        output, (h_n, c_n) = self.lstm(inputs)
        return output[:, -self.output_len_e1:, :]


class DecoderLSTM(nn.Module):
    def __init__(self, output_len_d, input_size_d, hidden_size_d, dropout_rate, output_size):
        super().__init__()
        self.output_len_d = output_len_d
        self.input_size_d = input_size_d
        self.hidden_size_d = hidden_size_d
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.lstm = nn.LSTM(input_size=self.input_size_d, hidden_size=self.hidden_size_d, num_layers=1, bias=True,
                            dropout=0, batch_first=True, bidirectional=False)
        self.out1 = nn.Linear(in_features=self.hidden_size_d, out_features=output_size, bias=True)

    def forward(self, inputs):
        output, (h_n, c_n) = self.lstm(inputs)
        output = self.dropout(output)
        final_output = self.out1(output)

        return final_output


# seq_len: 总序列长度
# output_len: 输出序列长度
# input_size: 输入层特征数量
# hidden_size: 输出层特征数量
# dropout_rate: 丢失率
class LSTM(nn.Module):
    def __init__(self, lstm_config: LSTMConfig):
        super().__init__()
        seq_len_e1 = lstm_config.seq_len_e
        output_len_e1 = lstm_config.output_len_e
        input_size_e1 = lstm_config.input_size_e
        hidden_size_e1 = lstm_config.hidden_size_e
        output_len_d = lstm_config.output_len_d
        input_size_d = lstm_config.hidden_size_e
        hidden_size_d = lstm_config.hidden_size_d
        output_size = lstm_config.output_size
        dropout_rate = lstm_config.dropout_rate

        self.encoder_obs = EncoderRNN1(seq_len_e1, output_len_e1, input_size_e1, hidden_size_e1)
        self.decoder = DecoderLSTM(output_len_d, input_size_d, hidden_size_d, dropout_rate, output_size)

    def forward(self, seq_x, seq_y_past):
        # 输入气象时间序列变量（n+m天），得到后m天的气象时间序列code
        decoder_inputs = self.encoder_obs(seq_x)
        # 输入流量时间序列变量（n天），得到流量code
        output = self.decoder(decoder_inputs)

        return output

    def get_last_layer(self):
        return self.decoder.out1
