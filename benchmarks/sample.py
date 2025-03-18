# import numpy as np
# import builtins

# # Create a temporary patch for np.bool
# np.bool = builtins.bool
# np.object = builtins.object

import warnings

import crypten
import crypten.communicator as comm
import crypten.mpc as mpc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.serialization import safe_globals
from torchvision import datasets, transforms

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

crypten.init()
torch.set_num_threads(1)

ALICE = 0
BOB = 1

mnist = datasets.MNIST(
    "/tmp", download=True, train=True, transform=transforms.ToTensor()
)
mean, std = (
    mnist.data.float().mean().unsqueeze(0),
    mnist.data.float().std().unsqueeze(0),
)
mnist_norm = transforms.functional.normalize(mnist.data.float(), mean, std)
mnist_labels = mnist.targets

data = mnist_norm[:20]
labels = mnist_labels[:20]

torch.save(data[:10], "/tmp/alice_train.pth")
torch.save(data[10:], "/tmp/bob_test.pth")
torch.save(labels[:10], "/tmp/alice_train_labels.pth")
torch.save(labels[10:], "/tmp/bob_test_labels.pth")


class AliceNet(nn.Module):
    def __init__(self):
        super(AliceNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


crypten.common.serial.register_safe_class(AliceNet)


def compute_accuracy(output, labels):
    pred = output.argmax(1)
    correct = pred.eq(labels)
    correct_count = correct.sum(0, keepdim=True).float()
    accuracy = correct_count.mul_(100.0 / output.size(0))
    return accuracy


# Load pre-trained model to Alice
dummy_model = AliceNet()
with safe_globals([AliceNet, Linear]):
    plaintext_model = torch.load("models/tutorial4_alice_model.pth", weights_only=False)

print(plaintext_model)

# Encrypt the model from Alice:

# 1. Create a dummy input with the same shape as the model input
dummy_input = torch.empty((1, 784))

# 2. Construct a CrypTen network with the trained model and dummy_input
private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)

# 3. Encrypt the CrypTen network with src=ALICE
private_model.encrypt(src=ALICE)

# Check that model is encrypted:
print("Model successfully encrypted:", private_model.encrypted)


# Classify encrypted data with encrypted model
labels = torch.load("/tmp/bob_test_labels.pth").long()
count = 100  # For illustration purposes, we'll use only 100 samples for classification


@mpc.run_multiprocess(world_size=2)
def encrypt_model_and_data():
    # Load pre-trained model to Alice
    model = crypten.load_from_party("models/tutorial4_alice_model.pth", src=ALICE)

    # Encrypt model from Alice
    dummy_input = torch.empty((1, 784))
    private_model = crypten.nn.from_pytorch(model, dummy_input, skip_onnx=True, remove_batch_norm=True)
    private_model.encrypt(src=ALICE)

    # Load data to Bob
    data_enc = crypten.load_from_party("/tmp/bob_test.pth", src=BOB)
    data_enc2 = data_enc[:count]
    data_flatten = data_enc2.flatten(start_dim=1)

    # Classify the encrypted data
    private_model.eval()
    output_enc = private_model(data_flatten)

    # Compute the accuracy
    output = output_enc.get_plain_text()
    accuracy = compute_accuracy(output, labels[:count])
    crypten.print("\tAccuracy: {0:.4f}".format(accuracy.item()))


encrypt_model_and_data()
