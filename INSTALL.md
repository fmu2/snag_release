# Requirements

- Linux
- Python >=3.6, <=3.10
- PyTorch 1.10+
- TensorBoard
- CUDA 10.2+
- GCC 4.9+
- NumPy 1.11+
- PyYaml
- torchtext (pip install torchtext)

# Compilation

Part of NMS is implemented in C++. The code can be compiled by

```shell
cd ./libs/utils
python setup_nms.py install --user
cd ../..
```

The code should be recompiled every time you update PyTorch.
