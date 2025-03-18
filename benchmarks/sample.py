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

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

crypten.init()
torch.set_num_threads(1)

ALICE = 0
BOB = 1

mnist = datasets.MNIST(
    "/tmp", download=True, train=False, transform=transforms.ToTensor()
)
mean, std = (
    mnist.data.float().mean().unsqueeze(0),
    mnist.data.float().std().unsqueeze(0),
)
mnist_norm = transforms.functional.normalize(mnist.data.float(), mean, std)
mnist_labels = mnist.targets

data = mnist_norm[:200]
labels = mnist_labels[:200]

torch.save(data, "/tmp/alice_train.pth")
torch.save(data, "/tmp/bob_test.pth")
torch.save(labels, "/tmp/alice_train_labels.pth")
torch.save(labels, "/tmp/bob_test_labels.pth")


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
    # Both parties create the same model architecture
    dummy_model = AliceNet()
    dummy_input = torch.empty((1, 784))

    # Load the model weights on Alice's side only
    # This will be a no-op for Bob

    with safe_globals([AliceNet, Linear]):
        model_data = torch.load(
            "models/tutorial4_alice_model.pth", weights_only=False
        )
        dummy_model.load_state_dict(model_data.state_dict())


    # Convert to CrypTen model
    private_model = crypten.nn.from_pytorch(dummy_model, dummy_input)

    # Encrypt the model with Alice as the source
    private_model.encrypt(src=ALICE)

    # Load test data - each party attempts to load their part
    test_data = None
    try:
        test_data = torch.load("/tmp/bob_test.pth")
    except:
        # Alice doesn't have the test data, create dummy data
        test_data = torch.zeros((count, 28, 28))

    # Encrypt the test data with Bob as the source
    data_enc = crypten.cryptensor(test_data[:count], src=BOB)

    # Flatten the encrypted data
    data_flatten = data_enc.flatten(start_dim=1)

    # Classify the encrypted data
    private_model.eval()
    output_enc = private_model(data_flatten)
    crypten.print("Output tensor encrypted:", crypten.is_encrypted_tensor(output_enc))

    # Get plaintext output
    output = output_enc.get_plain_text()
    pred = output.argmax(dim=1)
    crypten.print("Decrypted labels:\n", pred)

    # Both parties try to compute accuracy, but only Bob has the labels
    try:
        test_labels = torch.load("/tmp/bob_test_labels.pth").long()
        accuracy = compute_accuracy(output, test_labels[:count])
        crypten.print("Accuracy: {:.4f}".format(accuracy.item()))
    except:
        pass  # Alice doesn't have the labels


encrypt_model_and_data()
