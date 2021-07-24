from loader import ChallengeDataset
from torch.utils.data import DataLoader
from models import *
from trainer import trainer
import os

NUM_CLASSES=8
def task1():
    rep_type='point' # 'voxel','point' or 'spectral'

    path_model='model_objs/%s_model.pt'%rep_type

    dataset_train=ChallengeDataset('dataset','train',rep_type)
    loader_train = DataLoader(dataset_train, batch_size=32,shuffle=True,pin_memory=True,num_workers=4)
    dataset_test=ChallengeDataset('dataset','test',rep_type)
    loader_test = DataLoader(dataset_test, batch_size=32,shuffle=True,pin_memory=True,num_workers=4)

    if os.path.isfile(path_model):
        net = torch.load(path_model)
        print('Training from saved model')
    else:
        net = point_model(NUM_CLASSES) # voxel_model(n) or point_model(n) or spectral_model(n)
        print('Training from scratch')
    ch_trainer=trainer(net,loader_train,loader_test,rep_type)
    ch_trainer.train(100)

def task2():
    rep_type='voxel' # 'voxel','point' or 'spectral'
    path_model='model_objs/%s_model.pt'%rep_type

    dataset_train=ChallengeDataset('dataset','train',rep_type)
    loader_train = DataLoader(dataset_train, batch_size=32,shuffle=True,pin_memory=True,num_workers=4)
    dataset_test=ChallengeDataset('dataset','test',rep_type)
    loader_test = DataLoader(dataset_test, batch_size=32,shuffle=True,pin_memory=True,num_workers=4)

    if os.path.isfile(path_model):
        net = torch.load(path_model)
        print('Training from saved model')
    else:
        net = voxel_model(NUM_CLASSES) # voxel_model(n) or point_model(n) or spectral_model(n)
        print('Training from scratch')

    ch_trainer=trainer(net,loader_train,loader_test,rep_type)
    ch_trainer.train(100)

def task3():
    rep_type='spectral' # 'voxel','point' or 'spectral'
    path_model='model_objs/%s_model.pt'%rep_type

    dataset_train=ChallengeDataset('dataset','train',rep_type)
    loader_train = DataLoader(dataset_train, batch_size=32,shuffle=True,pin_memory=True,num_workers=4)
    dataset_test=ChallengeDataset('dataset','test',rep_type)
    loader_test = DataLoader(dataset_test, batch_size=32,shuffle=True,pin_memory=True,num_workers=4)

    if os.path.isfile(path_model):
        net = torch.load(path_model)
        print('Training from saved model')
    else:
        net = spectral_model(NUM_CLASSES) # voxel_model(n) or point_model(n) or spectral_model(n)
        print('Training from scratch')

    ch_trainer=trainer(net,loader_train,loader_test,rep_type)
    ch_trainer.train(80)

if __name__=='__main__':
    #task1() #run for task1
    #task2() #run for task2
    task3() #run for task3
