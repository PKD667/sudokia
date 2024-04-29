import gen
import numpy as np
import torch

from model import Sudokai

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("model.pth").to(device)

grid = gen.gen_grid()
# gen one puzzle
puzzle = gen.gen_puzzle(grid)

# solve the puzzle
puzzle = torch.from_numpy(puzzle).float().to(device)

# set the model to evaluation mode
model.eval()

# predict the solution
solution = model(puzzle)

# convert the solution to a numpy array
solution = solution.cpu().detach().numpy()
solution = np.argmax(solution, axis=2) + 1
solution = solution.reshape((9, 9))


# print the puzzle
print(puzzle.cpu().detach().numpy().astype(int))

# print the solution
print(solution)

# print the grid

print(grid)





