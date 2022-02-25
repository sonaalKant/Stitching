from __future__ import print_function, division
import os
import glob
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import random
import pickle
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def read_resize(imname, shape):
    img = cv2.imread(imname)
    img = cv2.resize(img, shape)
    return img

# pts are circular -> (x1,y1), (x2,y1), (x2,y2), (x1,y2)
def crop_patch(img, pts):
    pts = np.rint(pts).astype(np.int32)
    x1,y1 = pts[0]
    x2,y2 = pts[2]
    return img[y1:y2, x1:x2, :]

def get_rect_pts(center, size):
    centerx, centery = center
    pts = list()
    pts.append([centerx - size//2, centery - size//2])
    pts.append([centerx + size//2, centery - size//2])
    pts.append([centerx + size//2, centery + size//2])
    pts.append([centerx - size//2, centery + size//2])

    return np.array(pts).astype(np.float32)


def getPatchPoint(img):
    #img size is assumed to be 320, 240
    centerx = random.randint(100,220)
    centery = random.randint(100, 140)
    return np.array([centerx, centery])

def getHomo(pts):
    if len(pts.shape) == 1:
        pts = np.expand_dims(pts, 0)
    ones = np.ones((len(pts), 1))
    return np.concatenate([pts, ones], axis=-1)
def warp_points(pts, H):
    pts = getHomo(pts)
    ptsB = H@pts.T
    ptsB = ptsB[0:2] / ptsB[-1]
    return ptsB.T

class HomographyDataset(Dataset):

    def __init__(self, dirpath, generate=True, transform=None, name="train"):
        self.transform = transform
        self.transform2 = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5)),
                        ])
        
        imnames = glob.glob(dirpath + "/*.jpg")
        if generate:
            self.info = list()
            num = 500000 if name=="train" else 5000
            for i in range(num):
                if i % 1000 == 0:
                    print("Completed ", i)
                IAimname = random.choice(imnames)
                IA = read_resize(IAimname, (320,240))
                centerA = getPatchPoint(IA)
                ptsA = get_rect_pts(centerA, 128)
                error = np.random.randint(-32, 32, size=(4,2)).astype(np.float32)
            
                H_AB = cv2.getPerspectiveTransform(ptsA, ptsA + error)
                H_BA = np.linalg.inv(H_AB)
                IB = cv2.warpPerspective(IA, H_BA, (320, 240))
                centerB = warp_points(centerA, H_BA)
                centerB = centerB[0]
                ptsB = get_rect_pts(centerB, 128)
                    
                # im1 = cv2.circle(IA, (int(centerA[0]), int(centerA[1])), 2, (0,255,0), -1)
                # im2 = cv2.circle(IB, (int(centerB[0]), int(centerB[1])), 2, (0,255,0), -1)
                # p1 = crop_patch(IA, ptsA)
                # p2 = crop_patch(IB, ptsB)
                # cv2.imwrite("im1.jpg", im1)
                # cv2.imwrite("im2.jpg", im2)
                # cv2.imwrite("p1.jpg", p1)
                # cv2.imwrite("p2.jpg", p2)

                self.info.append([IAimname, ptsA, error, H_AB, H_BA, ptsB, centerA, centerB])
            pickle.dump(self.info, open(f"/vulcanscratch/sonaalk/Stitching/Phase2/Data/homography_data_{name}.pkl", "wb"))
        
        else:
            self.info = pickle.load(open(f"/vulcanscratch/sonaalk/Stitching/Phase2/Data/homography_data_{name}.pkl", "rb"))

        if name == "train":
           self.info = self.info[:10000]

    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, idx):
        IAimname, ptsA, error, H_AB, H_BA, ptsB, centerA, centerB = self.info[idx]
        IA = read_resize(IAimname, (320,240))
        IB = cv2.warpPerspective(IA, H_BA, (320,240))
        pA = crop_patch(IA, ptsA)
        pB = crop_patch(IB, ptsB)
        # try:
        gt = error / 32.

        pA = self.transform(pA)
        pB = self.transform(pB)
        X = torch.cat([pA,pB], axis=0)
        # except:
        #     import pdb;pdb.set_trace()
            
        return X, gt, ptsA, self.transform2(IA)


if __name__ == '__main__':

    T = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        ])

    H = HomographyDataset('/vulcanscratch/sonaalk/Stitching/P1TestSet/Phase2/', generate=True, transform=T, name="test")
    l = list()
    for i in range(0, len(H)):
        print(i, len(l))
        try:
            H.__getitem__(i)
        except:
            l.append(i)
    

    import pdb;pdb.set_trace()

    # X, gt = H.__getitem__(0)
    # p1 = X[:,:,0]
    # p2 = X[:,:,1]
