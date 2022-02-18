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

class HomographyModel(torch.nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()

        self.model = torch.nn.Sequential(*[
            torch.nn.Conv2d(2,64,3,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(64,64,3,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64,128,3,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,3,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128,128,3,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,3,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

        ])

        self.regress = torch.nn.Sequential(*[ 
            torch.nn.AdaptiveMaxPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 1024),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024,8) ])
    
    def forward(self, x):
        out = self.model(x)
        return self.regress(out)


class TensorDLT(torch.nn.Module):
    def __init__(self):
        super(TensorDLT, self).__init__()
    
    def forward(self, xA, xB):
        # expecting normalized points
        x1 = xA[:,:,0]
        y1 = xA[:,:,1]
        x2 = xB[:,:,0]
        y2 = xB[:,:,1]
        ones = torch.ones_like(x1)
        zeros = torch.zeros_like(x1)
        ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2*x1, x1*x2], axis=-1)
        ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, y2*x1, x1*x2], axis=-1)
        A = torch.cat([ax,ay], axis=-1)

        

        return 

if __name__ == '__main__':

    model = HomographyModel()
    inp = torch.randn(1,2,128,128)
    out = model(inp)
    print(out.shape)