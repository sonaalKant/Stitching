#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

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
# Add any python libraries here

def overlay(base, warp):
	# for i in range(warp.shape[0]):
	# 	for j in range(warp.shape[1]):
	# 		if base[i,j].any() > 0:
	# 			warp[i,j] = base[i,j]
	# return warp
	# import pdb;pdb.set_trace()
	warp[(base > 0).any()] = base
	return warp

def pano_size(imgs, H):
	shape = imgs[0].shape
	corner = np.array([[0,0,1],[shape[1]-1,0,1],[0,shape[0]-1,1],[shape[1], shape[0],1]])
	points = []
	for i in range(len(H)):
		for j in range(corner.shape[0]):
			tran_H = H[i]@corner[j].T
			tran_H = tran_H/tran_H[2]
			# import pdb; pdb.set_trace()
			points.append([tran_H[0], tran_H[1]])
	points = np.array(points)
	x_max, x_min = np.max(points[:,0]), np.min(points[:,0])
	y_max, y_min = np.max(points[:,1]), np.min(points[:,1])

	return abs(y_max-y_min).astype(int), abs(x_max-x_min).astype(int), x_min.astype(int), y_min.astype(int)

def create_pano(imgs, H):
	height, width, x_min, y_min = pano_size(imgs, H)
	iter_img = imgs.copy()
	anchor = iter_img.pop(len(imgs)//2)
	I = np.eye(3)
	I[0,2] += -x_min
	I[1,2] += -y_min
	width = 5000
	height = 5000
	pano = cv2.warpPerspective(anchor, I, (width, height))

	for i in range(len(H)):
		H[i][0,2] += -x_min
		H[i][1,2] += -y_min
		warped = cv2.warpPerspective(iter_img[i], H[i], (width, height))
		pano = overlay(pano, warped)
	return pano

def get_center_points(size, shape):
	H,W,_ = shape
	pts = np.array([(W/2-size/2, H/2-size/2), (W/2+size/2, H/2-size/2), (W/2+size/2, H/2+size/2), (W/2-size/2, H/2+size/2)])
	return pts

def main(ModelPath, BasePath, ModelType):
	# Add any Command Line arguments here
	# Parser = argparse.ArgumentParser()
	# Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
	
	# Args = Parser.parse_args()
	# NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""
	imnames = glob.glob(f"{BasePath}/*.jpg")
	imgs = [cv2.imread(im) for im in imnames]

	T = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
		transforms.CenterCrop(128),
		transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        ])

	shape = imgs[0].shape
	patches = [T(img) for img in imgs]


	"""
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

	if ModelType == 'Sup':
		model = HomographyModel().cuda()
	elif ModelType == 'Unsup':
		model = HomographyModelUnsupervised().cuda()

	model.load_state_dict(torch.load(ModelPath))

	anchor_idx = len(patches)//2

	ptsA = get_center_points(128, shape)
	ptsA = torch.from_numpy(ptsA).unsqueeze(0).cuda()

	H_list = list()

	for i in range(0, anchor_idx):
		print(i, i+1)
		pA = patches[i].unsqueeze(0)
		pB = patches[i+1].unsqueeze(0)
		input = torch.cat([pA,pB], axis=1).cuda()

		if ModelType == 'Sup':
			error = model(input)
		if ModelType == 'Unsup':
			pA, pB = torch.chunk(input, dim=1, chunks=2)
			_, error = model(input, ptsA, pA)
		
		error = error.cpu().detach().numpy()
		error = error.reshape(4,2)
		error = error * 32
		pts = ptsA.cpu().numpy().astype(np.float32)
		H = cv2.getPerspectiveTransform(pts, pts + error)
		H_list.append(H)
	
	for i in range(len(patches)-1, anchor_idx, -1):
		print(i, i-1)
		pA = patches[i].unsqueeze(0)
		pB = patches[i-1].unsqueeze(0)
		input = torch.cat([pA,pB], axis=1).cuda()

		if ModelType == 'Sup':
			error = model(input)
		if ModelType == 'Unsup':
			pA, pB = torch.chunk(input, dim=1, chunks=2)
			_, error = model(input, ptsA, pA)
		
		error = error.cpu().detach().numpy()
		error = error.reshape(4,2)
		error = error * 32
		pts = ptsA.cpu().numpy().astype(np.float32)
		H = cv2.getPerspectiveTransform(pts, pts + error)
		H_list.append(H)
	
		
	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	img = create_pano(imgs, H_list)
	cv2.imwrite("viz/mypano.png", img)
	
if __name__ == '__main__':
		# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ModelPath', dest='ModelPath', default='/vulcanscratch/sonaalk/Stitching/Phase2/Checkpoints/supervised_large_normalized/checkpint_17.pt', help='Path to load latest model from, Default:ModelPath')
	Parser.add_argument('--BasePath', dest='BasePath', default='/vulcanscratch/sonaalk/Stitching/P1TestSet/Phase1/TestSet3', help='Path to load Data')
	Parser.add_argument('--ModelType', dest='ModelType', default='Sup', help='Use Sup or Unsup')
	Args = Parser.parse_args()
	ModelPath = Args.ModelPath
	BasePath = Args.BasePath
	ModelType = Args.ModelType

	main(ModelPath, BasePath, ModelType)
 
