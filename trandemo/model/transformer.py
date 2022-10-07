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

        mask=mask.unsqueeze(1)

        oh = self.attention(qh, kh, vh, mask)

        o = self.combine_heads(oh)
        output = self.o_transform(o)

        return output

    def attention(self, q, k, v, mask=None):
        d_k = k.shape[-1]
        scores = paddle.matmul(q, paddle.transpose(k,(0,1,3,2))) / math.sqrt(d_k)
        if mask is not None:
            scores = self.masked_fill(scores, mask, -1e9)
        attn = F.softmax(scores)
        return paddle.matmul(attn, v)

    def masked_fill(self, x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, x, y)

    def split_heads(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        hidden_size = x.shape[2]
        x = x.reshape((batch_size, seq_len, self.head, hidden_size // self.head))
        x = paddle.transpose(x, (0,2, 1,3))
        return x

    def combine_heads(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        hidden_size = x.shape[-1]
        x = paddle.transpose(x, (0,2,1,3))
        x = x.reshape((batch_size, seq_len, hidden_size * self.head))
        return x


class ResConnection(nn.Layer):
    def __init__(self, hidden_size, p=0.1):
        super(ResConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class DecoderLayer(nn.Layer):
    def __init__(self, head, hidden_size, ):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(head, hidden_size)
        self.crs_attn = MultiheadAttention(head, hidden_size)
        self.ffn = FFNLayer(hidden_size, hidden_size)

        self.self_res_connect = ResConnection(hidden_size)
        self.crs_res_connect = ResConnection(hidden_size)
        self.ffn_res_connect = ResConnection(hidden_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.self_res_connect(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.crs_res_connect(x, lambda x: self.crs_attn(x, memory, memory, src_mask))
        x = self.ffn_res_connect(x, self.ffn)
        return x


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
        position = paddle.arange(0, max_len).unsqueeze(1).astype('float32')
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
    def __init__(self, dec_num, hidden_size):
        super(Decoder, self).__init__()
        self.layers = nn.LayerList(DecoderLayer(hidden_size=hidden_size, head=hidden_size) for _ in range(dec_num))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, memory, src_mask,tgt, tgt_mask):
        for layer in self.layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)
        return self.norm(tgt)


class Encoder(nn.Layer):
    def __init__(self, enc_num, head, hidden_size):
        super(Encoder, self).__init__()
        self.layers = nn.LayerList(EncoderLayer(head, hidden_size) for _ in range(enc_num))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Transformer(nn.Layer):
    def __init__(self, src_vocab, tgt_vocab, hidden_size, max_len, enc_num, dec_num, head):
        super(Transformer, self).__init__()

        self.encEmb = nn.Embedding(embedding_dim=hidden_size, num_embeddings=len(src_vocab))
        self.decEmb = nn.Embedding(embedding_dim=hidden_size, num_embeddings=len(tgt_vocab))

        self.posEnc = PositionalEncoding(hidden_size, max_len)

        self.encoder = Encoder(enc_num=enc_num, head=head, hidden_size=hidden_size)
        self.decoder = Decoder(dec_num=dec_num, hidden_size=hidden_size)

        self.output_transform = nn.Linear(hidden_size, len(tgt_vocab))

    def forward(self, src, tgt, pad=0):
        """
        :param pad:
        :param src: [batch_size,len]
        :param tgt:
        :return:
        """
        tgt = tgt[:, :-1]

        src_mask = (src != pad).unsqueeze(-2)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = self.get_seq_mask(tgt_mask) & tgt_mask

        x = self.encode(src, src_mask)
        x = self.decode(x, src_mask, tgt, tgt_mask)
        result = F.log_softmax(self.output_transform(x),axis=-1)
        return result

    def encode(self, src, src_mask):
        x = self.encEmb(src)
        x = self.posEnc(x)
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.decEmb(tgt)
        x = self.posEnc(x)
        return self.decoder(memory, src_mask, x, tgt_mask)

    # @staticmethod
    # def ger_src_mask(x, pad=0):
    #     """
    #     :param pad:
    #     :param x: padded_input [batch_size,len,hidden_size]
    #     :return:
    #     """
    #     # pad的位置为1
    #     mask = paddle.where(x != pad, 0, 1)
    #     return mask

    @staticmethod
    def get_seq_mask(x):
        """
        :param x: padded_input [batch_size,len]
        :return:
        """
        len_ = x.shape[-1]
        attn_shape = (1, len_, len_)
        subsequent_mask = paddle.triu(paddle.ones(attn_shape), diagonal=1)
        return subsequent_mask == 0

    @staticmethod
    def base_setting():
        config_dict = {
            "batch_size": 4096,
            "hidden_size": 512,
            "head": 8,
            "enc_num": 6,
            "dec_num": 6,
            "max_len": 128
        }
        return config_dict
