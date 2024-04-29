import torch.nn as nn
import torch.nn.functional as F

# Model
class Sudokai(nn.Module):
    def __init__(self):
        super(Sudokai, self).__init__()
        self.conv1 = nn.Conv1d(9, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*9, 81*9, bias=True)

    def forward(self, x):
        x = x.view(-1, 9, 9)  # reshape the input to match the expected input of Conv1d
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.fc1(x.view(x.size(0), -1))  # flatten the output before passing it to the fully connected layer
        return F.log_softmax(x, dim=1).view(-1, 81, 9)