
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class trainer():
    def __init__(self,model,loader_train,loader_test,rep_type,report_size=100):
        self.model=model
        self.loader_train=loader_train
        self.loader_test=loader_test
        self.report_size=report_size
        self.rep_type=rep_type
        self.ifgpu=torch.cuda.is_available()
        if self.ifgpu:
            self.model=self.model.cuda()
        
        self.criterion = nn.CrossEntropyLoss()
        self.mb_size=loader_train.batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)#, betas=(0.9, 0.999))
        
        report_folder='runs/3dclassification_'+self.rep_type
        
        #tensorboard writing info comment if not needed
        if os._exists(report_folder):
            shutil.rmtree(report_folder)
        self.writer= SummaryWriter(report_folder)
        
    

    def train(self,epoch_size):
        for epoch in range(epoch_size):
            print('Starting epoch %d'%(epoch+1))
            self.train_epoch(epoch)
            self.val_epoch(epoch)
            torch.save(self.model,'model_objs/%s_model.pt'%self.rep_type)
            

    def train_epoch(self,epoch):
        self.model.train()
        losssum_ep = 0
        losssum_rep = 0
        report_ctr = 0
        samples_sum=0

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.loader_train)):
            self.optimizer.zero_grad()
            if self.ifgpu:
                inputs=inputs.to(device='cuda:0')
                targets = targets.to(device='cuda:0')

            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            losssum_ep += loss.item()
            losssum_rep += loss.item()
            report_ctr += 1
            samples_sum +=self.mb_size
            self.optimizer.step()
            if batch_idx % self.report_size == (self.report_size - 1):
                print('Loss report: %.7f' % ( (losssum_rep) / (self.report_size*self.mb_size)))
                
                losssum_rep=0

        self.writer.add_scalar('training loss',losssum_ep / samples_sum ,epoch)
        print('Finished training with overall loss: %.7f' % (losssum_ep / samples_sum))
        return losssum_ep / samples_sum



    def val_epoch(self,epoch):
        self.model.eval()
        losssum_ep = 0
        samples_sum=0
        corrects=0
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.loader_test)):
            if self.ifgpu:
                inputs=inputs.to(device='cuda:0')
                targets = targets.to(device='cuda:0')
            
            outputs = self.model(inputs)
            corrects += targets.eq(torch.max(outputs,1)[1]).sum().item()
            loss = self.criterion(outputs, targets)
            losssum_ep+=loss.item()
            samples_sum+=self.mb_size

        print('Finished validation with overall loss: %.7f' % (losssum_ep / (samples_sum)))
        self.writer.add_scalar('validation loss',losssum_ep / samples_sum ,epoch)
        print('Accuracy: %.2f percent' % (corrects*100 / (samples_sum)))
        self.writer.add_scalar('accuracy',corrects*100 /  samples_sum ,epoch)
        return (losssum_ep / samples_sum)


