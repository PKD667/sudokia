import numpy as np
import torch
from model import Sudokai

def evaluate_model(dataset_path='dataset.npy'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("model.pth").to(device)
    dataset = np.load(dataset_path, allow_pickle=True)
    
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    model.eval()
    
    for puzzle, solution in dataset:
        puzzle_tensor = torch.from_numpy(puzzle).float().to(device)
        solution_tensor = torch.from_numpy(solution).long().to(device)
        
        output = model(puzzle_tensor)
        pred = output.argmax(dim=1, keepdim=True)
        
        correct_predictions += pred.eq(solution_tensor.view_as(pred)).sum().item()
        total_predictions += solution_tensor.nelement()
        
        loss = torch.nn.functional.nll_loss(output, solution_tensor, reduction='sum').item()
        total_loss += loss
    
    accuracy = 100. * correct_predictions / total_predictions
    average_loss = total_loss / total_predictions
    
    print(f'Accuracy: {accuracy}%')
    print(f'Average loss: {average_loss}')

if __name__ == "__main__":
    evaluate_model()
