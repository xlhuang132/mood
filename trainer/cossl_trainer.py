
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn
import argparse 
from utils.misc import AverageMeter  
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np 
import models 
import time 
import torch.optim as optim 
import os   
import datetime
import torch.nn.functional as F  
from .base_trainer import BaseTrainer

from loss.contrastive_loss import *
from utils import FusionMatrix
from models.projector import  Projector 

class FixMatch_Loss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)
        return Lx, Lu


class CoSSLTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)      
        self.train_criterion=FixMatch_Loss()
        if self.warmup_enable:
            self.rebuild_unlabeled_dataset_enable=True
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)  
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter()  
        self.mask_prob = AverageMeter()
        self.total_c = AverageMeter()
        self.used_c = AverageMeter()
        
    def train_step(self,pretraining=False):
        if self.pretraining:
            return self.train_warmup_step()
        else:
            return self.train_cossl_step()
    
    def train_cossl_step(self):
        self.model.train()
        loss =0
        # DL  
        try:        
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        if  isinstance(inputs_x,list)  :
            inputs_x=inputs_x[0]
        
        # DU   
        try:       
            data = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data = self.unlabeled_train_iter.next()
        inputs_u=data[0][0]
        inputs_u2=data[0][1]
        inputs_u3=data[0][2]
        gt_targets_u=data[1]
        
        batch_size = inputs_x.size(0) 
        num_class=self.num_classes
        # Transform label to one-hot
        targets_x2 = torch.zeros(batch_size, self.num_classes).scatter_(1, targets_x.view(-1,1), 1)
        
        inputs_x, targets_x2 = inputs_x.cuda(), targets_x2.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3  = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()
        
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            outputs_u = self.model(inputs_u)
            targets_u = torch.softmax(outputs_u, dim=1)

            max_p, p_hat = torch.max(targets_u, dim=1)
            select_mask = max_p.ge(self.conf_thres).float()

            total_acc = p_hat.cpu().eq(gt_targets_u).float().view(-1)
            if select_mask.sum() != 0:
                self.used_c.update(total_acc[select_mask != 0].mean(0).item(), select_mask.sum())
            self.mask_prob.update(select_mask.mean().item())
            self.total_c.update(total_acc.mean(0).item())

            p_hat = torch.zeros(p_hat.size(0), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
            select_mask = torch.cat([select_mask, select_mask], 0)

        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x2, p_hat, p_hat], dim=0)

        all_outputs = self.model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = self.train_criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu 

        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(Lx.item(), inputs_x.size(0))
        self.losses_u.update(Lu.item(), inputs_x.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.avg,self.losses_x.avg,self.losses_u.avg))
        return 
    
    