#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Sonaal Kant (sonaal@cs.umd.edu)
MS Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import sys
import os
from Network.Network import HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import argparse
from Dataset.dataCreation import HomographyDataset
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb

# Don't generate pyc codes
sys.dont_write_bytecode = True

def PrettyPrint(NumEpochs, MiniBatchSize, NumTrainSamples, NumValSamples):
	"""
	Prints all stats with all arguments
	"""
	print('Number of Epochs Training will run for ' + str(NumEpochs))
	print('Mini Batch Size ' + str(MiniBatchSize))
	print('Number of Training Images ' + str(NumTrainSamples))
	print('Number of Validation Images ' + str(NumValSamples))          

	
def TrainOperation(DirNamesTrain, DirNamesVal, NumEpochs, MiniBatchSize, CheckPointPath,
				  ModelType):
	"""
	Inputs: 
	DirNamesTrain - Variable with Subfolder paths to train files
	TrainLabels - Labels corresponding to Train/Test
	NumTrainSamples - length(Train)
	ImageSize - Size of the image
	NumEpochs - Number of passes through the Train data
	MiniBatchSize is the size of the MiniBatch
	SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
	CheckPointPath - Path to save checkpoints/model
	DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
	LatestFile - Latest checkpointfile to continue training
	BasePath - Path to COCO folder without "/" at the end
	LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
	Outputs:
	Saves Trained network in CheckPointPath and Logs to LogsPath
	""" 
	transform =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        ])
	
	train_dataset = HomographyDataset(DirNamesTrain, generate=False, transform=transform, name="train")
	val_dataset = HomographyDataset(DirNamesVal, generate=False, transform=transform, name="val")

	train_dataloader = DataLoader(train_dataset, batch_size=MiniBatchSize,
                        shuffle=True, num_workers=4)
	val_dataloader = DataLoader(val_dataset, batch_size=MiniBatchSize,
                        shuffle=True, num_workers=4)
	
	PrettyPrint(NumEpochs, MiniBatchSize, len(train_dataset), len(val_dataset))

	model = HomographyModel().cuda()

	criterion = nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

	for i in range(NumEpochs):
		model.train()
		train_loss = 0.
		for idx, (input, H_gt) in enumerate(train_dataloader):
			input = input.cuda()
			H_gt = H_gt.cuda()
			H_pred = model(input)
			H_gt = H_gt.view(H_gt.shape[0],-1)
			loss = criterion(H_pred, H_gt)
			train_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if idx % 10 == 0:
				train_loss_avg = train_loss / (idx+1)
				with torch.no_grad():
					model.eval()
					val_loss = 0.
					for idx1, (input, H_gt) in enumerate(val_dataloader):
						input = input.cuda()
						H_gt = H_gt.cuda()
						H_pred = model(input)
						H_gt = H_gt.view(H_gt.shape[0],-1)
						loss = criterion(H_pred, H_gt)
						val_loss += loss.item()
					
					val_loss_avg = val_loss / (idx1 +1)

					print(f"Epoch : {i}, Iter : {idx}, Train Loss : {train_loss_avg}, Val Loss : {val_loss_avg}")
					
					wandb.log({'train_loss' : train_loss_avg, 'val_loss' : val_loss_avg, 'epoch' : i })
				
				# train_loss = 0.
				model.train()
	
		if not os.path.isdir(CheckPointPath):
			os.makedirs(CheckPointPath)
		
		torch.save(model.state_dict(), os.path.join(CheckPointPath, f"checkpint_{i}.pt"))
		

def main():
	"""
	Inputs: 
	None
	Outputs:
	Runs the Training and testing code based on the Flag
	"""
	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--train_dir', default='/vulcanscratch/sonaalk/Stitching/Phase2/Data/Train', help='Base path of train images')
	Parser.add_argument('--val_dir', default='/vulcanscratch/sonaalk/Stitching/Phase2/Data/Val', help='Base path of val images')
	Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
	Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--MiniBatchSize', type=int, default=512, help='Size of the MiniBatch to use, Default:1')
	Parser.add_argument('--RunName', type=str, default="dummy", help='Name of run')


	Args = Parser.parse_args()
	NumEpochs = Args.NumEpochs
	train_dir = Args.train_dir
	val_dir = Args.val_dir
	MiniBatchSize = Args.MiniBatchSize
	CheckPointPath = os.path.join(Args.CheckPointPath, Args.RunName)
	ModelType = Args.ModelType

	wandb.init(project="Stitching", entity="sonaalk")
	wandb.run.name = Args.RunName
	wandb.config.update(Args)
	
	TrainOperation(train_dir, val_dir, NumEpochs, MiniBatchSize, CheckPointPath,
				  ModelType)
		
	
if __name__ == '__main__':
	main()
 
