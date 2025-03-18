import crypten
import crypten.communicator as comm
import crypten.mpc as mpc
import torch

crypten.init()
# Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)

CLIENT = 0
SERVER = 1


@mpc.run_multiprocess(world_size=2)
def run():
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


run()
