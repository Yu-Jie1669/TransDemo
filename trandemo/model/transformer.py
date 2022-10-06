import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class FFNLayer(nn.Layer):
    def __init__(self, hidden_size, inner_size, dropout=0.1):
        super(FFNLayer, self).__init__()
        self.input_transform = nn.Linear(hidden_size, inner_size)
        self.output_transform = nn.Linear(hidden_size, inner_size)
        self.dropout = dropout

    def forward(self, x):
        hidden_state = self.input_transform(x)
        return self.output_transform(F.dropout(F.relu(hidden_state), self.dropout))


class MultiheadAttention(nn.Layer):
    def __init__(self, head, hidden_size):
        super(MultiheadAttention, self).__init__()
        self.head = head
        assert hidden_size % self.head == 0, "head_num can not be split."
        self.q_transform = nn.Linear(hidden_size, hidden_size)
        self.k_transform = nn.Linear(hidden_size, hidden_size)
        self.v_transform = nn.Linear(hidden_size, hidden_size)
        self.o_transform = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask):
        q = self.q_transform(query)
        k = self.k_transform(key)
        v = self.v_transform(value)

        qh = self.split_heads(q)
        kh = self.split_heads(k)
        vh = self.split_heads(v)

        oh = self.attention(qh, kh, vh, mask)

        o = self.combine_heads(oh)
        output = self.o_transform(o)

        return output

    def attention(self, q, k, v, mask=None):
        d_k = k.shape[-1]
        scores = paddle.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = self.masked_fill(scores, mask, -1e9)
        attn = F.softmax(scores)
        return paddle.matmul(attn, v)

    def masked_fill(self, x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, y, x)

    def split_heads(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        hidden_size = x.shape[2]
        x = x.reshape((batch_size, seq_len, self.head, hidden_size // self.head))
        x = x.transpose((1, 2))
        return x

    def combine_heads(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        hidden_size = x.shape[-1]
        x = x.transpose((1, 2))
        x = x.reshape((batch_size, seq_len, hidden_size * self.head))
        return x


class ResConnection(nn.Layer):
    def __init__(self, hidden_size):
        super(ResConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, sublayer):
        return self.norm(x + sublayer(x))


class DecoderLayer(nn.Layer):
    def __init__(self):
        super(DecoderLayer, self).__init__()

    def forward(self, *inputs, **kwargs):
        pass


class EncoderLayer(nn.Layer):
    def __init__(self, head, hidden_size):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(head, hidden_size)
        self.ffn = FFNLayer(hidden_size, hidden_size)
        self.attn_res_connect = ResConnection(hidden_size)
        self.ffn_res_connect = ResConnection(hidden_size)

    def forward(self, x, mask):
        x = self.attn_res_connect(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.ffn_res_connect(x, self.ffn)
        return x


class PositionalEncoding(nn.Layer):
    def __init__(self, hidden_size, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len

        pe = paddle.zeros((max_len, hidden_size))
        # position: [[0],[1],[2]...[max_len-1]]
        position = paddle.arange(0, max_len).unsqueeze(1)
        # div_term: (shape: [hidden_size//2,] )
        div_term = paddle.exp(paddle.arange(0, hidden_size, 2) *
                              -(math.log(10000.0) / hidden_size))

        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        # pe: (shape: [1,max_len,hidden_size])
        pe = pe.unsqueeze(0)
        # register_buffer 作用：不会更新参数，但是保存模型时会作为模型的一部分保存下来
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_data):
        """
        :param input_data: [batch_size,len,hidden_size]
        :return:
        """
        input_data = input_data + self.pe[:, :input_data.shape[1]]
        return self.dropout(input_data)


class Decoder(nn.Layer):
    def __init__(self, dec_num):
        super(Decoder, self).__init__()
        self.layers = nn.LayerList(DecoderLayer() for _ in range(dec_num))

    def forward(self, *inputs, **kwargs):
        pass


class Encoder(nn.Layer):
    def __init__(self, enc_num):
        super(Encoder, self).__init__()
        self.layers = nn.LayerList(EncoderLayer() for _ in range(enc_num))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transformer(nn.Layer):
    def __init__(self, src_vocab, tgt_vocab, hidden_size, max_len, enc_num, dec_num):
        super(Transformer, self).__init__()

        self.encEmb = nn.Embedding(embedding_dim=512, num_embeddings=len(src_vocab))
        self.decEmb = nn.Embedding(embedding_dim=512, num_embeddings=len(tgt_vocab))

        self.posEnc = PositionalEncoding(hidden_size, max_len)

        self.encoder = Encoder(enc_num=enc_num)
        self.decoder = Decoder()

        self.output_transform = nn.Linear(hidden_size, hidden_size)

    def forward(self, src, tgt):
        """
        :param src:
        :param tgt:
        :return:
        """
        src_mask = self.getMask(src)
        tgt_mask = self.getMask(tgt)
        x = self.encoder(src, src_mask)
        x = self.decoder(x, src_mask, tgt, tgt_mask)
        proj = F.softmax(self.output_transform(x))
        return proj

    def encode(self, src, src_mask):
        x = self.encEmb(src)
        x = self.posEnc(x)
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.decEmb(tgt)
        x = paddle.concat((paddle.zeros((x.shape[0], 1, x.shape[-1])), x[:, 1:, :]), axis=1)
        x = self.posEnc(x)
        return self.decoder(memory, src_mask, x, tgt_mask)

    @staticmethod
    def getMask(x, pad=0):
        """
        :param pad:
        :param x: padded_input [batch_size,len,hidden_size]
        :return:
        """
        # pad的位置为1
        mask = paddle.where(x != pad, 0, 1)
        return mask

    @staticmethod
    def base_setting():
        config_dict = {

        }
