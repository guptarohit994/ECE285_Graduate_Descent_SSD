import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = torch.Tensor(n_channels)
        if scale ==None:
            self.gamma = None
        else:
            self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(self.n_channels)
        init.constant_(self.weight,self.gamma)
        
    def forward(self,x):
        x = torch.div(x,torch.norm(x,p=None,dim=1,keepdim=True)+self.eps)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
        

