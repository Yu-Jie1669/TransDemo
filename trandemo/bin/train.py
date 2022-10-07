import time

import paddle.optimizer
import paddle
import paddle.nn as nn
from tqdm import tqdm

from trandemo.model.transformer import Transformer
from trandemo.data.pipeline import DataPipeline
from trandemo.data.vocabulary import Vocabulary

src_path = "../../data/corpus.tc.32k.zh.shuf"
tgt_path = "../../data/corpus.tc.32k.en.shuf"

src_vocab = "../../data/vocab.32k.zh.txt"
tgt_vocab = "../../data/vocab.32k.en.txt"

tgt_vocab = Vocabulary(tgt_vocab, '<unk>', None, '<eos>', '<pad>')
src_vocab = Vocabulary(src_vocab, '<unk>', None, '<eos>', '<pad>')

base_config = Transformer.base_setting()
model = Transformer(src_vocab, tgt_vocab, base_config['hidden_size'], enc_num=base_config['enc_num'],
                    dec_num=base_config['dec_num'], max_len=base_config['max_len'], head=base_config['head'])


dataset = DataPipeline.getTrainDataset(src_path, tgt_path, src_vocab, tgt_vocab, base_config['batch_size'],
                                       base_config['max_len'])

optimizer = paddle.optimizer.Adam(parameters=model.parameters())
loss_compute = nn.CrossEntropyLoss()

for i, data in enumerate(dataset):
    src, tgt = data
    tgt_y = tgt[:, 1:]
    t = time.time()
    result = model(src, tgt)
    print("iter %d ok, cost %.3f seconds" % (i, time.time() - t))
