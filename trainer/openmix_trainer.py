from .base_trainer import BaseTrainer
from models.classifier  import Classifier
import torch
import torch.nn.functional as F
import math
import numpy as np
from utils.misc import AverageMeter  

class OpenMixTrainer(BaseTrainer):
    def __init__(self, cfg):

        super().__init__(cfg)
        # self.new_fc = Classifier()
        # self.froze_backbone_iter = cfg.ALGORITHM.OPENMIX.FROZE_BACKBONE_ITER
        self.theta1 = cfg.ALGORITHM.OPENMIX.THETA1
        self.theta2 = cfg.ALGORITHM.OPENMIX.THETA2
        self.alpha = cfg.ALGORITHM.OPENMIX.MIXUP_ALPHA
        self.lambda1 = cfg.ALGORITHM.OPENMIX.LAMBDA1
        self.lambda2 = cfg.ALGORITHM.OPENMIX.LAMBDA2
        self.init_loss()

    def init_loss(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_ppl = AverageMeter()
        self.losses_pll = AverageMeter()
        self.losses_opm = AverageMeter()


    def train_step(self, pretraining=False):
        if pretraining:
            return self.train_warmup_step()
        else :
            return self.train_openmix_step()


    def train_openmix_step(self):
        self.model.train()
        loss = 0

        # DL  
        try:
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 

        # DU 
        try:                 
            data = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data = self.unlabeled_train_iter.next()
        inputs_u=data[0][0]
        inputs_u2=data[0][1]
        ul_y=data[1] 

        inputs_x, targets_x = inputs_x.cuda(), targets_x.long().cuda(non_blocking=True)
        
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
        x=torch.cat((inputs_x,inputs_u,inputs_u2),dim=0)

        # 只训练新分类头 new_fn
        # if self.iter <= self.froze_backbone_iter:
        #     self.model.froze_backbone()
        #     self.model.froze_classifier()
        # else :
        #     self.model.unfroze_backbone()
        #     self.model.unfroze_classifier()

        feature = self.model(x,return_encoding=True)
        num_labels=inputs_x.size(0)
        l_feature, u_feature = feature[:num_labels], feature[num_labels:]
        # l_feature, u_feature = torch.split(feature, num_labels)
        
        # DL
        l_logits = self.model.fc(l_feature) 
        lx=self.l_criterion(l_logits, targets_x.long()) 
        self.losses_x.update(lx.item(), inputs_x.size(0))
        score_result = self.func(l_logits)
        now_result = torch.argmax(score_result, 1)  

        # lppl
        # using all label data and unlabel data
        norm = torch.norm(feature, dim=1, p=2).view(1, -1)
        norm = torch.mm(norm.T, norm)
        S = torch.div(torch.mm(feature, feature.T), norm)

        W = torch.where(S > self.theta1, 1, 0)
        d_target_x = targets_x.view(-1)
        W_labels = torch.tensor([[j.equal(i) for j in d_target_x] for i in d_target_x]).cuda()
        W[:num_labels, :num_labels] = W_labels

        lppl = torch.mul(torch.log(S), W) + torch.mul(torch.log(1-S), 1-W)
        lppl = - torch.sum(lppl.view(-1)) / lppl.numel()
        self.losses_ppl.update(lppl.item(), inputs_u.size(0))

        # lpll
        u_logits = self.model.fc(u_feature).softmax(dim=1)
        u_logits_max, pred_label = u_logits.max(dim=1)
        # u_mask = torch.where(u_logits_max > self.theta2, float(1), float(0))
        u_mask = u_logits_max.ge(self.theta2).float()
        
        # lpll = - F.cross_entropy(u_feature, pred_label, weight=u_mask) / u_mask.view(-1).sum()
        lpll = self.ul_criterion(u_feature, pred_label, weight=u_mask) / u_mask.view(-1).sum()
        self.losses_pll.update(lpll.item(), inputs_u.size(0))

        # OpenMix
        l_targets_hot = torch.zeros([num_labels, self.num_classes], dtype=torch.int).cuda()
        l_targets_hot = l_targets_hot.scatter_(1, targets_x.view(-1, 1), 1)

        ul_targets_hot = torch.zeros([pred_label.size(0), self.num_classes], dtype=torch.int).cuda()
        ul_targets_hot = ul_targets_hot.scatter_(1, pred_label.view(-1, 1), 1)
        # l_x, ul_x = torch.split(x, num_labels)
        l_x, ul_x = x[:num_labels], x[num_labels:]
        m_x, m_t = self.mix_up(l_x, l_targets_hot, 
                               ul_x, ul_targets_hot)
        
        m_feature = self.model(m_x)
        
        lopm = torch.norm(m_t - m_feature, p=2, dim=1).sum() / (m_x.size(0) * self.num_classes)
        self.losses_opm.update(lopm.item(), inputs_u.size(0))

        l_u = lppl + self.lambda1 * lpll + self.lambda2 * lopm
        self.losses_u.update(l_u.item(), inputs_u.size(0))
        loss = lx + l_u

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_loss:{:>5.4f} Loss_x:{:>5.4f}  Loss_u:{:>5.4f} =='
                             .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,
                            self.losses.val,self.losses_x.avg,self.losses_u.val))
            self.logger.info('=================  Loss_ppl:{:>5.4f} Loss_pll:{:>5.4f} Loss_opm:{:>5.4f} ==============='
                             .format(self.losses_ppl.val, self.losses_pll.val, self.losses_opm.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
        

    def mix_up(self, l_images, l_targets, ul_images, ul_targets):                 
        with torch.no_grad():     
            len_l=l_images.size(0)
            len_aux=ul_images.size(0)
            if len_aux==0: 
                return l_images
            elif len_aux>len_l:
                ul_images=ul_images[:len_l]
            elif len_aux<len_l:
                extend_num= math.ceil(len_l/len_aux)
                tmp=[ul_images]*extend_num
                tmp_t=[ul_targets]*extend_num
                ul_images=torch.cat(tmp,dim=0)[:len_l]
                ul_targets=torch.cat(tmp_t,dim=0)[:len_l]

            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(lam, 1.0 - lam)
            rand_idx = torch.randperm(l_images.size(0)) 
            mixed_images = lam * l_images + (1.0 - lam) * ul_images[rand_idx]
            mixed_targets = lam * l_targets + (1.0 - lam) * ul_targets[rand_idx]
            return mixed_images, mixed_targets
        


