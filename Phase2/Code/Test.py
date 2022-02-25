#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

from ast import Mod
import sys
import os
from Network.Network import HomographyModel, HomographyModelUnsupervised, normalize
from Misc.MiscUtils import *
from Misc.DataUtils import *
import argparse
from Dataset.dataCreation import HomographyDataset
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import cv2


# Don't generate pyc codes
sys.dont_write_bytecode = True

def unnormalize(img):
	img = img.permute(1,2,0).cpu().numpy()
	img = img - img.min()
	img = img / img.max()
	return (img*255).astype(np.uint8)

def save_visualizations(Images, H_gt, H_pred, ptsA, save_path):
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	
	for j in range(Images.shape[0]):
		
		img = unnormalize(Images[j])
		
		base_pts= (ptsA[j]).cpu().numpy().reshape(-1,1,2).astype(np.int32)
		
		gt_pts= (H_gt[j]*32).cpu().numpy().reshape(-1,1,2).astype(np.int32)
		pred_pts= (H_pred[j]*32).cpu().numpy().reshape(-1,1,2).astype(np.int32)

		gt_pts = gt_pts + base_pts
		pred_pts = pred_pts + base_pts

		img = cv2.polylines(img.copy(), [gt_pts], True, (0,0,255), 2)
		img= cv2.polylines(img, [pred_pts], True, (255,0,0), 2)

		cv2.imwrite(f"{save_path}/{j}.jpg", img)


def TestOperation(ModelPath, ModelType, BasePath, MiniBatchSize):
	transform =  transforms.Compose([
	transforms.ToPILImage(),
	transforms.Grayscale(),
	transforms.ToTensor(),
	transforms.Normalize((0.5), (0.5)),
	])

	if ModelType == 'Sup':
		model = HomographyModel().cuda()
	elif ModelType == 'Unsup':
		model = HomographyModelUnsupervised().cuda()
	
	criterion = nn.MSELoss()

	model.load_state_dict(torch.load(ModelPath))

	for name in ["val", "test"]:
		test_dataset = HomographyDataset(BasePath, generate=False, transform=transform, name=name)

		test_dataloader = DataLoader(test_dataset, batch_size=MiniBatchSize,
							shuffle=False, num_workers=4)

		with torch.no_grad():
			model.eval()
			val_loss = 0.
			for idx1, (input, H_gt, ptsA, IA) in enumerate(test_dataloader):
				input = input.cuda()
				H_gt = H_gt.cuda()
				if ModelType == 'Sup':
					H_pred = model(input)
					H_gt = H_gt.view(H_gt.shape[0],-1)
				if ModelType == 'Unsup':
					ptsA = ptsA.cuda()
					pA, pB = torch.chunk(input, dim=1, chunks=2)
					_, H_pred = model(input, ptsA, pA)
				
				loss = criterion(H_pred, H_gt)
				val_loss += loss.item()
			
			save_visualizations(IA, H_gt, H_pred, ptsA, save_path=f"viz/{ModelType}/{name}")
			
			val_loss_avg = val_loss / (idx1 +1)
		
		print(f"{name} Dataset, Average Loss : {val_loss_avg}")
		

def main():
	"""
	Inputs: 
	None
	Outputs:
	Prints out the confusion matrix with accuracy
	"""

	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ModelPath', dest='ModelPath', default='/vulcanscratch/sonaalk/Stitching/Phase2/Checkpoints/supervised_large_normalized/checkpint_17.pt', help='Path to load latest model from, Default:ModelPath')
	Parser.add_argument('--BasePath', dest='BasePath', default='/vulcanscratch/sonaalk/Stitching/P1TestSet/Phase2/', help='Path to load Data')
	Parser.add_argument('--ModelType', dest='ModelType', default='Sup', help='Use Sup or Unsup')
	Args = Parser.parse_args()
	ModelPath = Args.ModelPath
	BasePath = Args.BasePath
	ModelType = Args.ModelType

	MiniBatchSize = 16

	TestOperation(ModelPath, ModelType, BasePath, MiniBatchSize)
	 
if __name__ == '__main__':
	main()
 
