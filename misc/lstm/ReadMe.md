# LSTM

循环神经网络，循环的含义是输入T个序列，每一个序列都和同一个LSTM/RNN前向传播，其中第一个LSTM/RNN前向传播另外输入的hidden_state=None，其余为上一序列前向完成后输入的序列，这样，序列的长度并不影响LSTM/RNN的大小，只是增加了前向传播的次数，通过这种在时序上的共享参数能够提升时序建模的泛化能力，同时减小模型的大小。

---
## 参考资料
- [lstm_gru.ipynb](https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb)，代码中有lstm和gru单元的简单实现便于理解，另外参考[ConvLSTM_Experiments](https://github.com/BachelorDog/ConvLSTM_Experiments.git)中有convlstm和convgru实现。
- [convlstm.py](https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py)，也可以参考该模块。
- [HAR-stacked-residual-bidir-LSTMs](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs)，其中的residual双向卷积图示非常简介清楚。
