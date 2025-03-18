# import numpy as np
# import builtins

# # Create a temporary patch for np.bool
# np.bool = builtins.bool
# np.object = builtins.object

import crypten
import torch
from torchvision import datasets, transforms 

crypten.init()
torch.set_num_threads(1)

mnist = datasets.MNIST("/tmp", download=True, train=True, transform=transforms.ToTensor())
mean, std = mnist.data.float().mean().unsqueeze(0), mnist.data.float().std().unsqueeze(0)
mnist_norm = transforms.functional.normalize(mnist.data.float(), mean, std)
mnist_labels = mnist.targets

data = mnist_norm[:20]
labels = mnist_labels[:20]

torch.save(data[:10], "/tmp/alice_train.pth")
torch.save(data[10:], "/tmp/bob_test.pth")
torch.save(labels[:10], "/tmp/alice_train_labels.pth")
torch.save(labels[10:], "/tmp/bob_test_labels.pth")
