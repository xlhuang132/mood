
from .base_trainer import BaseTrainer
from .daso_trainer import DASOTrainer
from .mood_trainer import MOODTrainer
from .fixmatch_trainer import FixMatchTrainer
from .mixmatch_trainer import MixMatchTrainer
from .crest_trainer import CReSTTrainer
from .supervised_trainer import SupervisedTrainer
from .pseudolabel_trainer import PseudoLabelTrainer
from .openmatch_trainer import OpenMatchTrainer 
from .ccssl_trainer import CCSSLTrainer
from .mtcf_trainer import MTCFTrainer 
from .fixmatchbcl_trainer import FixMatchBCLTrainer
from .ood_detector_trainer import OODDetectorTrainer
from .abc_trainer import ABCTrainer
from .cossl_trainer import CoSSLTrainer
from .openmix_trainer import OpenMixTrainer

def build_trainer(cfg):
    alg=cfg.ALGORITHM.NAME
    if alg=='MOOD':
        return MOODTrainer(cfg)
    elif alg=='baseline':
        return SupervisedTrainer(cfg)
    elif alg=='FixMatch':
        return FixMatchTrainer(cfg)
    elif alg=='FixMatchBCLTrainer':
        return FixMatchBCLTrainer(cfg)
    elif alg=='MixMatch':
        return MixMatchTrainer(cfg)
    elif alg=='CReST':
        return CReSTTrainer(cfg)
    elif alg=='DASO':
        return DASOTrainer(cfg)
    elif alg=='PseudoLabel':
        return PseudoLabelTrainer(cfg)
    elif alg=='OpenMatch':
        return OpenMatchTrainer(cfg)
    elif alg=='MTCF':
        return MTCFTrainer(cfg) 
    elif alg== 'CCSSL':
        return CCSSLTrainer(cfg) 
    elif alg=='OODDetect':
        return OODDetectorTrainer(cfg)
    elif alg=='ABC':
        return ABCTrainer(cfg)
    elif alg=='CoSSL':
        return CoSSLTrainer(cfg)
    elif alg=='openmix':
        return OpenMixTrainer(cfg)
    else:
        raise "The algorithm type is not valid!"