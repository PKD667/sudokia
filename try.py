import torch
import torch.nn.functional as F

from train import sudokloss,device
from gen import gen_grid,gen_puzzle

grid = gen_grid()
puzzle = gen_puzzle(grid)

grid = torch.tensor(grid, dtype=torch.long).to(device)
puzzle = torch.tensor(puzzle, dtype=torch.float32).to(device)

sm = F.log_softmax(puzzle)
print(sm)

base_loss = sudokloss(puzzle,grid)