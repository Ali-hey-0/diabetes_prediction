import torch
import torch.nn as nn     





class DiabetesNet(nn.Module):
    def __init__(self,input_dim):
        super(DiabetesNet,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,16), #first Hidden Layer
            nn.ReLU(),
            nn.Linear(16,8), #Second Hidden Layer
            nn.ReLU(),
            nn.Linear(8,1), # Last Hidden Layer 
            nn.Sigmoid()
        )
        
        
    def forward(self,x):
        return self.net(x)