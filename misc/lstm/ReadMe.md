# LSTM

循环神经网络，循环的含义是输入T个序列，每一个序列都和同一个LSTM/RNN前向传播，其中第一个LSTM/RNN前向传播另外输入的hidden_state=None，其余为上一序列前向完成后输入的序列，这样，序列的长度并不影响LSTM/RNN的大小，只是增加了前向传播的次数，通过这种在时序上的共享参数能够提升时序建模的泛化能力，同时减小模型的大小。

---
## 参考资料
- [lstm_gru.ipynb](https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb)
