import torch
import torch.nn as nn

# example
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    
    def forward(self, x):
        return self.model(x)

    def parameters_lr(self):
        params = [ 
            { 'params' : self.model.parameters(), 'lr': 1e-3 },
        ]
        return params
