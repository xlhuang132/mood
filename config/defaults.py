from yacs.config import CfgNode as CN

_C = CN()

# Model
_C.MODEL = CN()
_C.MODEL.NAME = "WRN"
_C.MODEL.WIDTH = 2
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.EMA_DECAY = 0.999
_C.MODEL.EMA_WEIGHT_DECAY = 0.0
_C.MODEL.WITH_ROTATION_HEAD = False
_C.MODEL.DUAL_HEAD_ENABLE=False
_C.MODEL.DUAL_HEAD_OUT_DIM=0
_C.MODEL.PROJECT_FEATURE_DIM=128
 

# Feature Queue for DASO and MOOD
_C.MODEL.QUEUE = CN()
_C.MODEL.QUEUE.MAX_SIZE = 256
_C.MODEL.QUEUE.FEAT_DIM = 128


# Losses
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.LABELED_LOSS = "CrossEntropyLoss"  
_C.MODEL.LOSS.UNLABELED_LOSS = "CrossEntropyLoss"
_C.MODEL.LOSS.UNLABELED_LOSS_WEIGHT = 1.0   
_C.MODEL.LOSS.WITH_SUPPRESSED_CONSISTENCY = False 


# Cross Entropy
_C.MODEL.LOSS.CROSSENTROPY = CN()
_C.MODEL.LOSS.CROSSENTROPY.USE_SIGMOID = False

_C.MODEL.OPTIMIZER = CN()
_C.MODEL.OPTIMIZER.TYPE = "SGD"
_C.MODEL.OPTIMIZER.BASE_LR = 0.03
_C.MODEL.OPTIMIZER.MOMENTUM = 0.9
_C.MODEL.OPTIMIZER.WEIGHT_DECAY = 1e-4 

_C.ALGORITHM.NAME = "Supervised"
_C.ALGORITHM.CONFIDENCE_THRESHOLD = 0.95
_C.ALGORITHM.CONS_RAMPUP_SCHEDULE = "exp"  # "exp" or "linear"
_C.ALGORITHM.CONS_RAMPUP_ITERS_RATIO = 0.4
# 

_C.ALGORITHM.PRE_TRAIN=CN()
_C.ALGORITHM.PRE_TRAIN.ENABLE=False
_C.ALGORITHM.PRE_TRAIN.WARMUP_EPOCH=0
_C.ALGORITHM.PRE_TRAIN.SimCLR=CN()
_C.ALGORITHM.PRE_TRAIN.SimCLR.ENABLE=False
_C.ALGORITHM.PRE_TRAIN.SimCLR.K=200
_C.ALGORITHM.PRE_TRAIN.SimCLR.TEMPERATURE=0.5
_C.ALGORITHM.PRE_TRAIN.SimCLR.FEATURE_DIM=64

_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR=CN()
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.TEMPERATURE=1. #   
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.THRESHOLD=0.95  #  
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.K=10  # top-k   
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.OOD_THRESHOLD=0.5 #  
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.ID_THRESHOLD=0.5 # 

# PseudoLabel
_C.ALGORITHM.PSEUDO_LABEL = CN()

# Mean Teacher
_C.ALGORITHM.MEANTEACHER = CN()
_C.ALGORITHM.MEANTEACHER.APPLY_DASO = False

# MixMatch
_C.ALGORITHM.MIXMATCH = CN()
_C.ALGORITHM.MIXMATCH.NUM_AUG = 2
_C.ALGORITHM.MIXMATCH.TEMPERATURE = 0.5
_C.ALGORITHM.MIXMATCH.MIXUP_ALPHA = 0.75
_C.ALGORITHM.MIXMATCH.APPLY_DASO = False

# FixMatch
_C.ALGORITHM.FIXMATCH = CN()

# DASO
_C.ALGORITHM.DASO = CN() 
_C.ALGORITHM.DASO.WARMUP_EPOCH=10
_C.ALGORITHM.DASO.PROTO_TEMP = 0.05
_C.ALGORITHM.DASO.PL_DIST_UPDATE_PERIOD = 100

# pseudo-label mixup
_C.ALGORITHM.DASO.WITH_DIST_AWARE = True
_C.ALGORITHM.DASO.DIST_TEMP = 1.0
_C.ALGORITHM.DASO.INTERP_ALPHA = 0.5
_C.ALGORITHM.DASO.WARMUP_ITER=5000
# prototype option
_C.ALGORITHM.DASO.QUEUE_SIZE = 256
# Semantic Alignment loss
_C.ALGORITHM.DASO.PSA_LOSS_WEIGHT = 1.0

# CReST
_C.ALGORITHM.CREST = CN()
_C.ALGORITHM.CREST.GEN_PERIOD_EPOCH = 50  
_C.ALGORITHM.CREST.ALPHA = 3.0
_C.ALGORITHM.CREST.TMIN = 0.5
_C.ALGORITHM.CREST.PROGRESSIVE_ALIGN = False

# OpenMatch
_C.ALGORITHM.OPENMATCH=CN() 
_C.ALGORITHM.OPENMATCH.LAMBDA_OEM=1.
_C.ALGORITHM.OPENMATCH.LAMBDA_SOCR=1. 
_C.ALGORITHM.OPENMATCH.MU=2. 
_C.ALGORITHM.OPENMATCH.START_FIX=10
_C.ALGORITHM.OPENMATCH.T=1.

# MOOD
_C.ALGORITHM.MOOD = CN()
_C.ALGORITHM.MOOD.NUM_AUG = 2
_C.ALGORITHM.MOOD.TEMPERATURE = 0.5
_C.ALGORITHM.MOOD.MIXUP_ALPHA = 0.75
_C.ALGORITHM.MOOD.BETA = 0.999 
_C.ALGORITHM.MOOD.FEATURE_LOSS_TEMPERATURE=1.
_C.ALGORITHM.MOOD.PAP_LOSS_WEIGHT=1.
 
# MTCF
_C.ALGORITHM.MTCF = CN()
_C.ALGORITHM.MTCF.MIXUP_ALPHA = 0.75 
_C.ALGORITHM.MTCF.LAMBDA_U=75
_C.ALGORITHM.MTCF.T=0.5    

# dataset
_C.DATASET = CN()
_C.DATASET.BUILDER = "build_cifar10_dataset"
_C.DATASET.NAME = "cifar10"  
_C.DATASET.ROOT = "./data"
#  
_C.DATASET.SAMPLER=CN()
_C.DATASET.SAMPLER.NAME = "RandomSampler"
_C.DATASET.SAMPLER.BETA = 0.999
#  
_C.DATASET.DUAL_SAMPLER=CN()
_C.DATASET.DUAL_SAMPLER.ENABLE=False
_C.DATASET.DUAL_SAMPLER.NAME="RandomSampler"

_C.DATASET.RESOLUTION = 32
_C.DATASET.BATCH_SIZE = 64 
_C.DATASET.NUM_WORKERS = 8
_C.DATASET.DOMAIN_DATASET_RETURN_INDEX=False
_C.DATASET.UNLABELED_DATASET_RETURN_INDEX=False
_C.DATASET.LABELED_DATASET_RETURN_INDEX=False
_C.DATASET.DL=CN()

_C.DATASET.GROUP_SPLITS=[3,3,4]
_C.DATASET.IFS=[100]
_C.DATASET.OODRS=[0.0]
 
_C.DATASET.IMB_TYPE='exp'

_C.DATASET.DL.NUM_LABELED_HEAD = 1500
_C.DATASET.DL.IMB_FACTOR_L = 100

_C.DATASET.DU=CN()
_C.DATASET.DU.TOTAL_NUM=10000
_C.DATASET.DU.UNLABELED_BATCH_RATIO=2
_C.DATASET.DU.ID=CN()
_C.DATASET.DU.ID.NUM_UNLABELED_HEAD = 3000
_C.DATASET.DU.ID.IMB_FACTOR_UL = 100
_C.DATASET.DU.ID.REVERSE_UL_DISTRIBUTION = False

_C.DATASET.DU.OOD=CN()
_C.DATASET.DU.OOD.ENABLE=False
_C.DATASET.DU.OOD.DATASET=''
_C.DATASET.DU.OOD.ROOT='./data'
_C.DATASET.DU.OOD.RATIO=0.0 #  
_C.DATASET.NUM_CLASSES=10
_C.SAVE_EPOCH=100
_C.MAX_EPOCH=500 
_C.VAL_ITERATION=500
_C.SHOW_STEP=20
_C.TRAIN_STEP=100 
 
# transform parameters
_C.DATASET.TRANSFORM = CN()
_C.DATASET.TRANSFORM.UNLABELED_STRONG_AUG = True
_C.DATASET.TRANSFORM.LABELED_STRONG_AUG = False
_C.DATASET.TRANSFORM.STRONG_AUG = True

_C.DATASET.NUM_VALID= 5000
_C.DATASET.REVERSE_UL_DISTRIBUTION=False
# analyse model 

_C.OUTPUT_DIR = "outputs"
_C.RESUME = '' 
_C.EVAL_ON_TEST_SET = True
_C.GPU_ID = 0
_C.MEMO = ""

# Reproducability
_C.SEED = 7
_C.SEED_PATH_ENABLE=False
_C.CUDNN_DETERMINISTIC = True
_C.CUDNN_BENCHMARK = False

_C.GPU_MODE=True
def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()