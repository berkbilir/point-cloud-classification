import torch
import torch.nn as nn
import torch.nn.functional as F


class point_model(nn.Module):
	### Complete for task 1
	def __init__(self,num_classes):
		super(point_model, self).__init__()
		
		self.mlp1=nn.Conv1d(3,64,1)
		self.mlp2 = nn.Conv1d(64,128,1)
		self.mlp3 = nn.Conv1d(128,1024,1)

		self.fc1 = nn.Linear(1024,512)
		self.fc2 = nn.Linear(512,256)
		self.fc3 = nn.Linear(256,num_classes)


	def forward(self,x):
		x = x.permute(0, 2, 1)
		x = F.relu(self.mlp1(x))
		x = F.relu(self.mlp2(x))
		x = F.relu(self.mlp3(x))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 1024)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x
		

class voxel_model(nn.Module):
	### Complete for task 2
	def __init__(self,num_classes):
		super(voxel_model, self).__init__()
		self.conv1=nn.Conv3d(1,8,3)
		self.maxpool1 = nn.MaxPool3d((2, 2, 2))
		self.conv2=nn.Conv3d(8,16,3)
		self.maxpool2 = nn.MaxPool3d((2, 2, 2))
		self.conv3=nn.Conv3d(16,32,3)
		self.maxpool3 = nn.MaxPool3d((2, 2, 2))
		self.flatten = nn.Flatten()

		self.fc1 = nn.Linear(256,16)
		self.fc2 = nn.Linear(16,num_classes)


	def forward(self,x):
		#x = x.permute(0, 2, 1)
		x = F.relu(self.conv1(x))
		x = self.maxpool1(x)
		x = F.relu(self.conv2(x))
		x = self.maxpool2(x)
		x = F.relu(self.conv3(x))
		x = self.maxpool3(x)
		x = self.flatten(x)
		#x = x.view(x.size(0), -1)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		return x
		

class spectral_model(nn.Module):
	### Complete for task 3
	def __init__(self,num_classes):
		super(spectral_model, self).__init__()
		self.mlp1 = nn.Conv1d(6,64,1)
		self.mlp2 = nn.Conv1d(64,128,1)
		self.mlp3 = nn.Conv1d(128,256,1)
		self.flatten = nn.Flatten()

		self.fc1 = nn.Linear(256,128)
		self.fc2 = nn.Linear(128,64)
		self.fc3 = nn.Linear(64,num_classes)

	def forward(self,x):
		#x = x.float()

		x = x.permute(0, 2, 1)
		x = F.relu(self.mlp1(x))
		x = F.relu(self.mlp2(x))
		x = F.relu(self.mlp3(x))
		x = torch.max(x, 2, keepdim=True)[0]
		x = self.flatten(x)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)


		return x


