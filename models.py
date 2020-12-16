import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0, std=1e-3)

def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1)
    T_norm = row_abs / row_sum
    return T_norm



class Matrix_optimize(nn.Module):
    def __init__(self, basis_num, num_classes):
        super(Matrix_optimize, self).__init__()
        self.basis_matrix = self._make_layer(basis_num, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-1)

    def _make_layer(self, basis_num, num_classes):
    
        layers = []
        for i in range(0, basis_num):
          layers.append(nn.Linear(num_classes, 1, False))
        return nn.Sequential(*layers)
        
    def forward(self, W, num_classes):
        results = torch.zeros(num_classes, 1)
        for i in range(len(W)):
            
            coefficient_matrix = float(W[i]) * torch.eye(num_classes, num_classes)
            self.basis_matrix[i].weight.data = norm(self.basis_matrix[i].weight.data) # s.t.
            anchor_vector = self.basis_matrix[i](coefficient_matrix)
            results += anchor_vector
            self.basis_matrix[i].weight.data = norm(self.basis_matrix[i].weight.data)
        return results
        

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.T_revision = nn.Linear(10, 10, False)

    def forward(self, x, revision=True):
        correction = self.T_revision.weight
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out_1 = F.relu(self.fc2(out)) # -> representations
        out_2 = self.fc3(out_1)
        if revision == True:
            return out_1, out_2, correction
        else:
            return out_1, out_2



