import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Sudokai

# load data from dataset.npy
dataset = np.load("dataset.npy", allow_pickle=True)

# initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize the model
model = Sudokai().to(device)

def sudokloss(output, target):
    distributed_target = []
    target = target.reshape(81)
    output = output.squeeze(0)
    for n in target:
        distribution = [0,0,0,0,0,0,0,0,0]
        distribution[n-1] = 1
        distributed_target.append(distribution)

    distributed_target = torch.tensor(distributed_target).to(device).float()

    #print(f"target : {distributed_target.shape} output : {output.shape}")
    #print(output.dtype)
    #print(distributed_target.dtype)
    # compute the loss between distributed target and output
    loss = F.mse_loss(distributed_target.flatten(),output.flatten())

    return loss

# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# define the training function
def sudoktrain():
    # set the model to training mode
    model.train()
    # iterate through the dataset
    for i, (puzzle, grid) in enumerate(dataset):
        # reshape the puzzle and grid
        puzzle = torch.tensor(puzzle, dtype=torch.float32).to(device)
        grid = torch.tensor(grid, dtype=torch.long).to(device)
        # get the model's prediction
        #print(puzzle.shape)
        output = model(puzzle) + 1
        # compute the loss
        loss = sudokloss(output, grid)
        # zero out the gradients
        optimizer.zero_grad()
        # perform backprop
        loss.backward()
        # update the weights
        optimizer.step()
        # print the loss every 100 iterations
        if i % 100 == 0:
            print("Iteration %d, loss = %.4f" % (i, loss.item()))
    
    # save the model
    torch.save(model, "model.pth")

if __name__ == "__main__":
    # train the model
    for i in range(10):
        sudoktrain()
        print("Epoch %d complete" % (i + 1))

                
            