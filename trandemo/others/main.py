import math

import paddle

import matplotlib.pyplot as plt
import numpy as np

max_len = 128
d_model = 20

pe = paddle.zeros((max_len, d_model))
pe1 = paddle.zeros((max_len, d_model))
position = paddle.arange(0, max_len,dtype='float32').unsqueeze(1)

sub1 = paddle.arange(0, d_model, 2)

div_term = paddle.exp(paddle.arange(0, d_model, 2) *
                      -(math.log(10000.0) / d_model))

div_term2 = paddle.exp(paddle.arange(0, d_model, 2)  * -math.log(10000.0)/ d_model )

pe[:, 0::2] = paddle.sin(position * div_term)
pe[:, 1::2] = paddle.cos(position * div_term)
pe = pe.unsqueeze(0)

pe1[:, 0::2] = paddle.sin(position * div_term2)
pe1[:, 1::2] = paddle.cos(position * div_term2)
pe1 = pe1.unsqueeze(0)

plt.figure(figsize=(15, 5))
y = paddle.zeros((1, 100, 20),dtype='float32')+pe1[:, :100]
plt.plot(np.arange(100), y[0, :, 4:8].numpy())
plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
plt.show()
