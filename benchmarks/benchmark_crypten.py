import warnings

import crypten
import crypten.mpc as mpc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.serialization import safe_globals

warnings.filterwarnings(
    "ignore", message="You are using `torch.load` with `weights_only=False`"
)
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

crypten.init()
# Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)

CLIENT = 0
SERVER = 1


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(12000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 71)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

crypten.common.serial.register_safe_class(MLP)


model = MLP()
torch.save(model, "models/mlp_local.pt")


@mpc.run_multiprocess(world_size=2)
def run():
    dummy_model = MLP()
    dummy_input = torch.empty((1, 12000))
    
    # Encrypt model
    with safe_globals([MLP, Linear]):
        model_data = torch.load("models/mlp_local.pt", weights_only=False)
        dummy_model.load_state_dict(model_data.state_dict())
    private_model = crypten.nn.from_pytorch(dummy_model, dummy_input)
    private_model.encrypt(src=SERVER)
    crypten.print(f"Model: {private_model}")

    # Encrypt data
    data = torch.load("models/input.pth")
    data_enc = crypten.cryptensor(data, src=CLIENT)

    # Encrypted inference
    private_model.eval()
    out_enc = private_model(data_enc)
    out = out_enc.get_plain_text()
    print(f"Output: {out}")


run()
