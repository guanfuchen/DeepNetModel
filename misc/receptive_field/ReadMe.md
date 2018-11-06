# 感受野计算

```
RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF= ((RF-1)* stride) + fsize
```

---
## 参考资料

- [Receptive-Field-in-Pytorch](https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/Receptive_Field.ipynb)
- ...
