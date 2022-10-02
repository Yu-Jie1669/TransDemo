import math

import paddle
import paddle.nn as nn


class FFNLayer(nn.Layer):
    def __init__(self):
        super(FFNLayer, self).__init__()

    def forward(self, *inputs, **kwargs):
        pass


class MultiheadAttention(nn.Layer):
    def __init__(self):
        super(MultiheadAttention, self).__init__()

    def forward(self, *inputs, **kwargs):
        pass


class DecoderLayer(nn.Layer):
    def __init__(self):
        super(DecoderLayer, self).__init__()

    def forward(self, *inputs, **kwargs):
        pass


class EncoderLayer(nn.Layer):
    def __init__(self):
        super(EncoderLayer, self).__init__()

    def forward(self, *inputs, **kwargs):
        pass


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
        # 作用：不会更新参数，但是保存模型时会作为模型的一部分保存下来
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
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, *inputs, **kwargs):
        pass


class Encoder(nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self):
        pass


class Transformer(nn.Layer):
    def __init__(self, src_vocab, tgt_vocab):
        super(Transformer, self).__init__()

        self.encEmb = nn.Embedding(embedding_dim=512, num_embeddings=len(src_vocab))
        self.decEmb = nn.Embedding(embedding_dim=512, num_embeddings=len(tgt_vocab))

        self.posEnc = PositionalEncoding()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input_data):
        pass


    @staticmethod
    def base_setting():
        config_dict={

        }
