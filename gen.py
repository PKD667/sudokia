# this file is supposed to generate pairs of unsolved and solved sudoku puzzles
# the format a list of pairs of matrices
# the first matrix is the unsolved puzzle with 0s in the empty spaces
# the second matrix is the solved puzzle


import random
import sys
import os
import time
import numpy as np

# this function is used to generate a solved sudoku (entire grid)
# it should respect the rules of sudoku
def gen_grid():
    # generate the first row
    row = [i for i in range(1, 10)]
    random.shuffle(row)
    grid = [row]
    # generate the rest of the rows
    for i in range(1, 9):
        row = grid[i - 1].copy()
        # shift the row by 3
        row = row[3:] + row[:3]
        # shift the row by 1
        row = row[1:] + row[:1]
        grid.append(row)

    # convert the grid to a numpy matrix
    grid = np.matrix(grid)
    return grid

# this function is used to generate a sudoku puzzle from a grid
def gen_puzzle(grid):
    # generate a random number of holes
    num_holes = random.randint(26, 36)
    # generate a list of random indices
    indices = []
    while len(indices) < num_holes:
        index = random.randint(0, 80)
        if index not in indices:
            indices.append(index)
    # generate the puzzle
    puzzle = grid.copy()
    for index in indices:
        puzzle[index // 9, index % 9] = 0
    return puzzle


def gen_dataset(num_pairs):
    # generate a list of pairs of unsolved and solved puzzles
    dataset = []
    for i in range(num_pairs):
        # generate a random grid
        grid = gen_grid()
        # generate a puzzle from the grid
        puzzle = gen_puzzle(grid)
        # append the pair to the dataset
        dataset.append((puzzle, grid))
    dataset = np.array(dataset)
    return dataset



if __name__ == "__main__":
    dataset = gen_dataset(1000)
    print(dataset.shape)
    # print a radom puzzle
    print(dataset[random.randint(0, 99)])
    # save the dataset
    np.save("dataset.npy", dataset)