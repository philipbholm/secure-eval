# import the libraries
import crypten
import crypten.communicator as comm
import crypten.mpc as mpc
import torch

# initialize crypten
crypten.init()
# Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)


@mpc.run_multiprocess(world_size=2)
def examine_arithmetic_shares():
    x_enc = crypten.cryptensor([1, 2, 3], ptype=crypten.mpc.arithmetic)

    rank = comm.get().get_rank()
    crypten.print(f"\nRank {rank}:\n {x_enc}\n", in_order=True)


x = examine_arithmetic_shares()
