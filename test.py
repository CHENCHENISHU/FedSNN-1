import torch
print(torch.__version__)

from pysnn.datasets import nmnist_train_test

# 替换为你要存储/读取N-MNIST数据集的本地路径
data_root = "D:/datasets/nmnist"

train_loader, test_loader = nmnist_train_test(root=data_root)
help(
    torch
)