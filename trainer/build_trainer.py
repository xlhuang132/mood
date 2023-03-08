
from .base_trainer import BaseTrainer
from .daso_trainer import DASOTrainer
from .mood_trainer import MOODTrainer
from .fixmatch_trainer import FixMatchTrainer
from .mixmatch_trainer import MixMatchTrainer
from .crest_trainer import CReSTTrainer
from .supervised_trainer import SupervisedTrainer  

def build_trainer(cfg):
    alg=cfg.ALGORITHM.NAME
    if alg=='MOOD':
        return MOODTrainer(cfg)
    elif alg=='Supervised':
        return SupervisedTrainer(cfg)
    elif alg=='FixMatch':
        return FixMatchTrainer(cfg)
    elif alg=='MixMatch':
        return MixMatchTrainer(cfg)
    elif alg=='CReST':
        return CReSTTrainer(cfg)
    elif alg=='DASO':
        return DASOTrainer(cfg)
    else:
        raise "The algorithm type is not valid!"