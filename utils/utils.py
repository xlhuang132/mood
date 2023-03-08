import numpy as np
import torch
import os
import copy
import sys 
 
def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1  
    offsets = interleave_offsets(batch, nu)  
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]  
    for i in range(1, nu + 1):  
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]  
    return [torch.cat(v, dim=0) for v in xy]  

def load_checkpoint(model,model_path,cfg=None):
    assert os.path.exists(model_path)        
    print(f"===> Loading checkpoint '{model_path}'")
    checkpoint = torch.load(model_path)               
    model.load_state_dict(checkpoint["model"])    
    return model

def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95

def print_results(results, in_dataset, out_dataset, name, method):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: ' + in_dataset)
    print('out_distribution: '+ out_dataset)
    print('Model Name: ' + name)
    print('')

    print(' OOD detection method: ' + method)
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
    print('')

def get_group_splits(cfg): 
    num_classes=cfg.DATASET.NUM_CLASSES
    group_splits=cfg.DATASET.GROUP_SPLITS
    assert num_classes==sum(group_splits)
    id_splits=[]
    l=0
    class_ids=np.array([i for i in range(num_classes)])
    for item in group_splits:
        id_splits.append(class_ids[l:l+item])
        l+=item
    return id_splits
   
def get_DL_dataset_path(cfg,dataset=None,):
    dataset = cfg.DATASET.NAME if not dataset else dataset
    path=os.path.join(cfg.OUTPUT_DIR, dataset)
    return path

def get_DL_dataset_alg_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None):
    parent_path=get_DL_dataset_path(cfg,dataset=dataset)
    algorithm_name=cfg.ALGORITHM.NAME   if not algorithm else algorithm 
    model_name=cfg.MODEL.NAME
    labeled_loss_type=cfg.MODEL.LOSS.LABELED_LOSS_CLASS_WEIGHT_TYPE if not labeled_loss_type else labeled_loss_type
    if labeled_loss_type and labeled_loss_type!='None':
        algorithm_name=algorithm_name+labeled_loss_type
    path=os.path.join(parent_path, algorithm_name, model_name)
    return path

def get_DL_dataset_alg_DU_dataset_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None,
             num_labeled_head=None,imb_factor_l=None,num_unlabeled_head=None,imb_factor_ul=None):
    parent_path=get_DL_dataset_alg_path(cfg,dataset=dataset,algorithm=algorithm,labeled_loss_type=labeled_loss_type)
    num_labeled_head=cfg.DATASET.DL.NUM_LABELED_HEAD if not num_labeled_head else num_labeled_head
    imb_factor_l=cfg.DATASET.DL.IMB_FACTOR_L if not imb_factor_l else imb_factor_l
    num_unlabeled_head=cfg.DATASET.DU.ID.NUM_UNLABELED_HEAD if not num_unlabeled_head else num_unlabeled_head
    imb_factor_ul=cfg.DATASET.DU.ID.IMB_FACTOR_UL  if not imb_factor_ul else imb_factor_ul
    DL_DU_ID_setting='DL-{}-IF-{}-DU{}-IF_U-{}'.format(num_labeled_head,imb_factor_l,num_unlabeled_head,imb_factor_ul)
    path=os.path.join(parent_path, DL_DU_ID_setting)
    return path

def get_DL_dataset_alg_DU_dataset_OOD_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None,
             num_labeled_head=None,imb_factor_l=None,num_unlabeled_head=None,imb_factor_ul=None,
             ood_dataset=None,ood_r=None
             ): 
    parent_path=get_DL_dataset_alg_DU_dataset_path(cfg,dataset=dataset,algorithm=algorithm,labeled_loss_type=labeled_loss_type,
             num_labeled_head=num_labeled_head,imb_factor_l=imb_factor_l,num_unlabeled_head=num_unlabeled_head,imb_factor_ul=imb_factor_ul)
    ood_dataset=cfg.DATASET.DU.OOD.DATASET if not ood_dataset else ood_dataset
    ood_r=cfg.DATASET.DU.OOD.RATIO if not ood_r else ood_r 
    OOD_setting='OOD-{}-r-{:.2f}'.format(ood_dataset,ood_r)
    path=os.path.join(parent_path, OOD_setting)
    return path
  
def get_root_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None,
             num_labeled_head=None,imb_factor_l=None,num_unlabeled_head=None,imb_factor_ul=None,
             ood_dataset=None,ood_r=None,
             sampler=None,sampler_mixup=None,dual_sampler_enable=None,dual_sampler=None,dual_sampler_mixup=None,
             Branch_setting=None,  
             ):
    parent_path=get_DL_dataset_alg_DU_dataset_OOD_path(cfg,dataset=dataset,algorithm=algorithm,labeled_loss_type=labeled_loss_type,
             num_labeled_head=num_labeled_head,imb_factor_l=imb_factor_l,
             num_unlabeled_head=num_unlabeled_head,imb_factor_ul=imb_factor_ul,
             ood_dataset=ood_dataset,ood_r=ood_r
             ) 
    if not Branch_setting:
        sampler=cfg.DATASET.SAMPLER.NAME if not sampler else sampler
        sampler_mixup=cfg.ALGORITHM.BRANCH1_MIXUP if sampler_mixup==None else sampler_mixup
        dual_sampler_enable=cfg.DATASET.DUAL_SAMPLER.ENABLE if dual_sampler_enable==None else dual_sampler_enable
        dual_sampler=cfg.DATASET.DUAL_SAMPLER.NAME if not dual_sampler else dual_sampler
        dual_sampler_mixup=cfg.ALGORITHM.BRANCH2_MIXUP if dual_sampler_mixup==None else dual_sampler_mixup
        if sampler_mixup:
            Branch_setting='{}_mixup'.format(sampler) 
        else:
            Branch_setting='{}'.format(sampler) 
        if dual_sampler_enable:
            if dual_sampler_mixup:
                Branch_setting+='-{}_mixup'.format(dual_sampler) 
            else:
                Branch_setting+='-{}'.format(dual_sampler)  
         
    path=parent_path
    if cfg.ALGORITHM.ABLATION.ENABLE:
        file_name=get_ablation_file_name(cfg)
        path=os.path.join(path,file_name) 
    return path 

def prepare_output_path(cfg,logger): 
    path= get_root_path(cfg)
    model_dir = os.path.join(path ,"models") 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        logger.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        ) 
    print("=> output model will be saved in {}".format(model_dir)) 
    pic_dir= os.path.join(path ,"pic") 
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir) 
    return path,model_dir,pic_dir 
 

def cal_metric(known, novel, method=None):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']

    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results
 
 