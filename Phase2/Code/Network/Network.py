"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision



Author(s):
Sonaal Kant (sonaal@cs.umd.edu)
MS Candidate in Computer Science,
University of Maryland, College Park
"""

import torch
import sys
import numpy as np
import torch
# Don't generate pyc codes
sys.dont_write_bytecode = True

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.network = torch.nn.Sequential(*[
            torch.nn.Conv2d(2,64,3,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(64,64,3,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64,128,3,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,3,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128,128,3,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,3,1),
            torch.nn.ReLU(),

        ])

        self.regress = torch.nn.Sequential(*[ 
            torch.nn.AdaptiveMaxPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 1024),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024,8) ])
    
    def forward(self, x):
        out = self.network(x)
        return self.regress(out)

if __name__ == '__main__':

    model = Network()
    inp = torch.randn(1,2,128,128)
    out = model(inp)
    print(out.shape)