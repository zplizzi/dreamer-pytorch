import os

from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import time


# Setup
results_dir = os.path.join("results", "bench3_{}".format(time.time()))
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda")


BATCH_SIZE = 10000
SIZE = 100

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(SIZE, SIZE)
        self.linear2 = nn.Linear(SIZE, SIZE)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)

mlp = MLP().to(device)

# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ],
#     on_trace_ready=torch.profiler.tensorboard_trace_handler(results_dir),
# ) as p:
data = torch.randn(BATCH_SIZE, SIZE).to(device)
# mlp = torch.jit.script(mlp)
mlp = torch.jit.trace_module(mlp, {"forward": data})
# # print(mlp.graph)
# print(mlp.code)
# print(mlp.graph_for(x))

for s in tqdm(range(50000)):
    mlp(data)
