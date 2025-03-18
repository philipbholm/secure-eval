import crypten
import crypten.mpc as mpc
import torch
import torch.nn as nn
from torch.serialization import safe_globals

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
        x = x * x
        x = self.fc2(x)
        x = x * x
        x = self.fc3(x)
        return x


@mpc.run_multiprocess(world_size=2)
def run():
    # Add MLP to safe globals to allow loading the model
    with safe_globals([MLP]):
        model = crypten.load_from_party("models/mlp.pt", src=SERVER)

    dummy_input = torch.empty((1, 12000))
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src=SERVER)

    # Load data from client
    data_enc = crypten.load_from_party("models/input.pth", src=CLIENT)

    private_model.eval()
    out_enc = private_model(data_enc)
    out = out_enc.get_plain_text()
    print(f"Output: {out}")


# run()
