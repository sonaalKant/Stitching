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
import pickle 
import torch.nn.functional as F
from kornia.geometry.epipolar.fundamental import normalize_points
import cv2
import warnings

def read_resize(imname, shape):
    img = cv2.imread(imname)
    img = cv2.resize(img, shape)
    return img


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

'''
This is similar to cv2.getPerspectiveTransform
'''
class TensorDLT(torch.nn.Module):
    def __init__(self):
        super(TensorDLT, self).__init__()
    
    def forward(self, xA, xB):
        
        eps = 1e-8

        xA, TA = normalize_points(xA)
        xB, TB = normalize_points(xB)

        # expecting normalized points
        x1, y1 = torch.chunk(xA, dim=-1, chunks=2)
        x2, y2 = torch.chunk(xB, dim=-1, chunks=2)
        
        ones = torch.ones_like(x1).cuda()
        zeros = torch.zeros_like(x1).cuda()
        # ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2*x1, y2*y1], axis=-1)
        # ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2*x1, -x2*y1], axis=-1)
        # A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

        # bb = torch.cat([-y2, x2], axis=-1).view(x1.shape[0],-1,1) 
        # Ap = torch.pinverse(A, rcond=1e-15) 

        ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2], dim=-1)
        ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1, -x2], dim=-1)
        A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])
        
        A = A + 1e-4*torch.randn(A.shape).cuda()

        try:
            _, S, V = torch.svd(A)
        except RuntimeError:
            warnings.warn('SVD did not converge', RuntimeWarning)
            return torch.empty((xA.size(0), 3, 3)).cuda


        # H = Ap @ bb # B,8,1
        # H = torch.cat([H, torch.ones(xA.shape[0],1,1).cuda()], axis=1) # B,9,1
        # H = H.view(H.shape[0],3,3)

        H = V[..., -1].view(-1, 3, 3)

        H = TB.inverse() @ (H @ TA)
        H = H / (H[..., -1:, -1:] + eps)

        return H

'''
This is similar to cv2.warpPerspective
'''
class STN(torch.nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        H = 240
        W = 320
        self.M = torch.tensor([[W/2, 0, W/2],[0, H/2, H/2],[0,0,1]]).float().cuda()
        self.M_inv = torch.inverse(self.M).float().cuda()

        self.affine_identity = torch.tensor([[1,0,0],[0,1,0]]).float().cuda()
    
    def forward(self, x, H):
        
        eps = 1e-8
        H_inv = torch.inverse(H).float()
        H_inv = self.M_inv.repeat(x.shape[0],1,1) @ (H_inv @ self.M.repeat(x.shape[0],1,1)) #B,3,3

        grid = F.affine_grid(self.affine_identity.repeat(x.shape[0],1,1), x.size()) # B,H,W,2

        B,h,w,_ = grid.shape

        ones = torch.ones(grid.shape[0:3]).unsqueeze(-1).cuda()
        grid = torch.cat([grid, ones], axis=-1) # B,H,W,3

        grid = grid.view(grid.shape[0],-1, grid.shape[-1]).permute(0,2,1) # B,3,H*W

        grid = H_inv@grid # B,3,H*W

        grid = grid.permute(0,2,1).view(B,h,w,3)

        grid = grid / (grid[:,:,:,-1:] + eps)

        grid = grid[:,:,:,:2]

        xB = F.grid_sample(x,grid)

        return xB

class HomographyModelUnsupervised(HomographyModel):
    def __init__(self):
        super(HomographyModelUnsupervised, self).__init__()
        self.dlt = TensorDLT()
        self.stn = STN()
    
    def forward(self, x, ptsA, patchA): 
        out = self.model(x)
        error = self.regress(out)
        error = error*32
        error = error.view(error.shape[0],-1,2)
        H = self.dlt(ptsA, ptsA+error)
        H_inv = torch.inverse(H)
        patchB = self.stn(patchA, H_inv)
        return patchB, error / 32.

def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img *255

if __name__ == '__main__':

    model = HomographyModel()
    inp = torch.randn(1,2,128,128)
    out = model(inp)
    print(out.shape)

    data = pickle.load(open("/vulcanscratch/sonaalk/Stitching/Phase2/Data/homography_data_val.pkl", "rb"))

    IAimname, ptsA, error, H_AB, H_BA, ptsB, centerA, centerB = data[0]


    ptsB = ptsA + error
    ptsA = torch.from_numpy(ptsA).unsqueeze(0)
    ptsB = torch.from_numpy(ptsB).unsqueeze(0)

    # Check Tensor DLT
    dlt = TensorDLT()
    out = dlt(ptsA, ptsB)
    print(out.shape)

    # check STN
    stn = STN()
    IA = read_resize(IAimname, (320,240))
    H = torch.from_numpy(H_BA).unsqueeze(0)
    img = torch.from_numpy(IA).unsqueeze(0).permute(0,3,1,2)
    img = img / img.max()
    out = stn(img, H)
    imgB = out[0].permute(1,2,0).numpy()
    cv2.imwrite("check.jpg",imgB*255)
    IB = cv2.warpPerspective(IA, H_BA, (320, 240))
    cv2.imwrite("check1.jpg", IB)

    # check Unsup
    model = HomographyModelUnsupervised().cuda()
    # inp = torch.randn(1,2,128,128)
    # pA = torch.randn(1,1,128,128)
    # ptsA = torch.randn(1,4,2)
    out = model(inp, ptsA, pA)
    import pdb;pdb.set_trace()
    
    print(out.shape)