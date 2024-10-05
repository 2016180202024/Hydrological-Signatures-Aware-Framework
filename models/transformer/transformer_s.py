import torch
import torch.nn as nn

from configs.model_config.TransformerS_config import TransformerSConfig
from configs.data_config.project_config import ProjectConfig

d_model = -999
n_heads = -999
n_encoder_layers = -999
n_decoder_layers = -999
d_ff = -999
dropout_rate = -999
d_head = -999
pred_len = -999
src_len = -999
tgt_len = -999
src_size = -999
tgt_size = -999
device = None


def set_global_value(transformer_config):
    global d_model, n_heads, n_encoder_layers, n_decoder_layers, d_ff, dropout_rate, d_head
    global pred_len, src_len, tgt_len, src_size, tgt_size, device
    d_model = transformer_config.d_model
    n_heads = transformer_config.n_heads
    n_encoder_layers = transformer_config.n_encoder_layers
    n_decoder_layers = transformer_config.n_decoder_layers
    d_ff = transformer_config.d_ff
    dropout_rate = transformer_config.dropout_rate
    d_head = transformer_config.d_head
    pred_len = transformer_config.pred_len
    src_len = transformer_config.src_len
    tgt_len = transformer_config.tgt_len
    src_size = transformer_config.src_size
    tgt_size = transformer_config.tgt_size
    device = ProjectConfig.device


# Subsequence Mask
def get_subsequence_mask(seq_len, pt=0):
    if pt == 0:
        subsequence_mask = torch.zeros((seq_len, seq_len)).to(device)
        subsequence_mask[:, -pred_len:] = float("-inf")
    elif pt == 1:
        # Upper triangular matrix
        subsequence_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(device)
    else:
        raise RuntimeError(f"Not such subsequence mask decode_mode:{pt}.")

    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask):
        """
        Q: [batch_size, n_heads, len_q, d_head)]
        K: [batch_size, n_heads, len_k(=len_v), d_head]
        V: [batch_size, n_heads, len_v(=len_k), d_head]
        mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(d_head, dtype=torch.float32))  # scores : [batch_size, n_heads, len_q, len_k]

        if mask is not None:
            scores += mask

        attn = torch.softmax(scores, dim=-1)
        # attn: [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V)
        # context: [batch_size, n_heads, len_q, d_head]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k(=len_v), d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_head).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_head).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_head).transpose(1, 2)

        context = ScaledDotProductAttention()(Q, K, V, mask)
        # context: [batch_size, n_heads, len_q, d_head]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_head)
        output = self.fc(context)
        # output: [batch_size, len_q, d_model]
        return output


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        output = self.feed_forward(inputs)
        return output


# EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.ffn = FeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, enc_layer_inputs, enc_self_attn_mask):
        """
        enc_layer_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_layer_inputs (Q, K, V is the same)
        residual1 = enc_layer_inputs.clone()
        enc_self_attn_outputs = self.enc_self_attn(enc_layer_inputs, enc_layer_inputs, enc_layer_inputs,
                                                   enc_self_attn_mask)
        outputs1 = self.norm1(enc_self_attn_outputs + residual1)

        residual2 = outputs1.clone()
        ffn_outputs = self.ffn(outputs1)
        # ffn_outputs: [batch_size, src_len, d_model]
        ffn_outputs = self.dropout(ffn_outputs)
        outputs2 = self.norm2(ffn_outputs + residual2)

        return outputs2


# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_encoder_layers)])

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = enc_inputs.clone()
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs


class TransformerS(nn.Module):
    def __init__(self, transformer_config: TransformerSConfig):
        super().__init__()
        set_global_value(transformer_config)
        self.src_pos_emb = nn.Embedding(src_len, d_model)
        # self.tgt_pos_emb = nn.Embedding(tgt_len, d_model)
        self.src_linear = nn.Linear(src_size, d_model)
        # self.tgt_linear = nn.Linear(tgt_size, d_model)

        self.encoder = Encoder()
        # self.decoder = Decoder()
        self.linear_project = nn.Linear(d_model, tgt_size)

        self.dropout = nn.Dropout(dropout_rate)

    # Transformer-Simple的输入：
    # src：气象时间序列+静态特征拼接
    # tgt：流量时间序列+静态签名拼接，气象时间对应前一天流量时间
    def forward(self, src, tgt):
        # Position Embedding and Input Projection
        batch_size = src.shape[0]
        src_position = torch.tensor(range(src_len)).unsqueeze(0).repeat(batch_size, 1).to(device)
        # tgt_position = torch.tensor(range(tgt_len)).unsqueeze(0).repeat(batch_size, 1).to(device)
        # src = torch.cat([src, tgt], 2)
        # src: [batch_size, src_len, src_size]
        src_inputs = self.src_pos_emb(src_position) + self.src_linear(src)
        # tgt_inputs = self.tgt_pos_emb(tgt_position) + self.tgt_linear(tgt)

        # Encoder
        enc_self_attn_mask = None
        enc_outputs = self.encoder(src_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size, src_len, d_model]

        # Decoder
        # dec_self_attn_mask = get_subsequence_mask(tgt_len, pt=0)  # dec_self_attn_mask: [tgt_len, tgt_len]
        # dec_enc_attn_mask = None
        # dec_outputs = self.decoder(tgt_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        # # dec_outputs: [batch_size, tgt_len, d_model]

        # Linear project
        # project_outputs = self.linear_project(dec_outputs)
        project_outputs = self.linear_project(enc_outputs)
        # project_outputs: [batch_size, tgt_len, tgt_size]

        return project_outputs

    def get_last_layer(self):
        return self.linear_project
