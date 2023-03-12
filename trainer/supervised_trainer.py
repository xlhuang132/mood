
 
import torch     
from .base_trainer import BaseTrainer 

class SupervisedTrainer(BaseTrainer):        
    
    def train_step(self,pretraining=False):
        self.model.train()
        loss =0
        # DL  
        try:        
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next()   
        
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)     
         
        logits_x = self.model(inputs_x) 
        # loss computation 
        loss=self.l_criterion(logits_x, targets_x.long())  
        
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)          
         
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(loss.item(), inputs_x.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.val,self.losses_x.avg,self.losses_u.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
    
    def get_val_model(self,):
        return self.model
