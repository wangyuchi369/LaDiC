import argparse
import random

import numpy as np
import torch
import math
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
from transformers import BertTokenizer, BertForMaskedLM

init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=120 * 60))
accelerator = Accelerator(log_with="wandb", kwargs_handlers=[init_kwargs])

# helper functions
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def save_model_tokenizer(tokenizer_class, model_class, name):
    if tokenizer_class is not None:
        tokenizer = tokenizer_class.from_pretrained(name)
        tokenizer.save_pretrained(f"./tokenizers/{name}-local")
    if model_class is not None:
        model = model_class.from_pretrained(name)
        model.save_pretrained(f"./models/{name}-local/")




# training hyperparameter


def loss_l2(x_hat, x):
    return (x_hat - x).square().mean()
def loss_l2_seq(x_hat, x):
    return torch.norm(x_hat - x, dim=-1).mean()
def loss_l3(x_hat, x):
    return torch.pow((((x_hat - x).abs()) ** 3).mean(), 1 / 3)
LOSS_FUNC = loss_l2_seq




parser = argparse.ArgumentParser()
parser.add_argument('--notes', type=str, default=None, help='Note to be included in the trial name')
parser.add_argument('--bsz', type=int, default=64, help='batch size')
parser.add_argument('--seqlen', type=int, default=24, help='sequence length')
parser.add_argument('--epoch', type=int, default=60, help='epoch num')
parser.add_argument('--resume_epoch', type=int, default=0, help='start epoch of resume')
parser.add_argument('--resume_ckpt', type=str, default=None, help='resume or not')
parser.add_argument('--logdir', type=str, default='checkpoint', help='logdir')
parser.add_argument('--var_dilate', type=float, default=4, help='variance dilate in diffusion')
parser.add_argument('--var_dilate_val', type=float, default=1, help='variance dilate in diffusion during validation')
parser.add_argument('--using_time', type=bool, default=True, help='using time or not')
parser.add_argument('--beta_min', type=float, default=0.0001, help='Minimum value of beta')
parser.add_argument('--beta_max', type=float, default=0.02, help='Maximum value of beta')
parser.add_argument('--step_tot', type=int, default=1000, help='Total noise adding steps')
parser.add_argument('--cosin_schedule', type=bool, default=False, help='Use cosine scheduling for alpha sequence')
parser.add_argument('--x_0_prediction', type=bool, default=True, help='Predict x_0 or x_{t-1}')
parser.add_argument('--use_x_t_loss', type=bool, default=True, help='Use x_t loss')
parser.add_argument('--use_x_1_loss', type=bool, default=False, help='Use x_1 loss')
parser.add_argument('--use_prob_loss', type=bool, default=True, help='Use probability loss')
# Argument Parsing


# Seed and Debugging
parser.add_argument('--seed', type=int, default=3407, help='Random seed')
# parser.add_argument('--resume', type=bool, default=False, help='Resume training')
parser.add_argument('--debug', type=bool, default=False, help='Debug mode')

# Training hyperparameters
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')

parser.add_argument('--early_stop_ratio', type=float, default=1.05, help='Early stop ratio')

# Dataset hyperparameters
parser.add_argument('--image_size', type=int, default=224, help='Image size')
parser.add_argument('--x_mean', type=str, default='datasets/mean_emb_split.pickle', help='Path to X mean')
parser.add_argument('--x_sigma', type=str, default='datasets/std_emb_split.pickle', help='Path to X sigma')

# Loss definition
parser.add_argument('--dynamic_rounding_weight', type=float, default=-1, help='Dynamic rounding weight')
parser.add_argument('--rounding_weight', type=float, default=0.2, help='Rounding weight')

parser.add_argument('--epsilon_pred', type=bool, default=False, help='Predict epsilon')
parser.add_argument('--self_cond', type=bool, default=True, help='Self condition')
parser.add_argument('--self_cond_prob', type=float, default=0.5, help='Self condition probability')
parser.add_argument('--classifier_free_weight', type=float, default=1, help='Classifier free weight')
parser.add_argument('--classifier_free_prob', type=float, default=0.1, help='Classifier free probability')
parser.add_argument('--train_embedding', type=bool, default=False, help='Train embedding')
parser.add_argument('--use_object', type=bool, default=False, help='Use object')
parser.add_argument('--pos_max', type=int, default=5, help='Max position')
parser.add_argument('--max_object_length', type=int, default=7, help='Max object length')
parser.add_argument('--object_mask_ratio', type=float, default=0.1, help='Object mask ratio')
parser.add_argument('--using_ln', type=bool, default=False, help='Using LN')
parser.add_argument('--using_time_ln', type=bool, default=False, help='Using time LN')
parser.add_argument('--use_early_proj', type=bool, default=False, help='Use early projection')

args = parser.parse_args()
notes = args.notes

# Extracting parsed arguments
SEED = args.seed
setup_seed(SEED)
# RESUME = args.resume
DEBUG = args.debug
LEARNING_RATE = args.learning_rate
WARMUP_RATIO = args.warmup_ratio
EARLY_STOP_RATIO = args.early_stop_ratio
IMAGE_SIZE = args.image_size
X_MEAN = torch.load(args.x_mean, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
X_SIGMA = torch.load(args.x_sigma, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
DYNAMIC_ROUNDING_WEIGHT = args.dynamic_rounding_weight
ROUNDING_WEIGHT = args.rounding_weight
EPSILON_PRED = args.epsilon_pred
SELF_COND = args.self_cond
SELF_COND_PROB = args.self_cond_prob
CLASSIFIER_FREE_WEIGHT = args.classifier_free_weight
CLASSIFIER_FREE_PROB = args.classifier_free_prob
TRAIN_EMBEDDING = args.train_embedding
IN_CHANNEL = 16 if TRAIN_EMBEDDING else 768
USE_OBJECT = args.use_object
POS_MAX = args.pos_max
MAX_OBJECT_LENGTH = args.max_object_length
OBJECT_MASK_RATIO = args.object_mask_ratio
USING_LN = args.using_ln
USING_TIME_LN = args.using_time_ln
USE_EARLY_PROJ = args.use_early_proj


BETA_MIN = args.beta_min
BETA_MAX = args.beta_max
STEP_TOT = args.step_tot
COSIN_SCHEDULE = args.cosin_schedule
X_0_PREDICTION = args.x_0_prediction
USE_X_T_LOSS = args.use_x_t_loss
USE_X_1_LOSS = args.use_x_1_loss
USE_PROB_LOSS = args.use_prob_loss
TRAIN_BATCH_SIZE = args.bsz
VAL_BATCH_SIZE = 200
# RESUME = args.resume
MAX_LENGTH = args.seqlen  # max text length
EPOCH_NUM = args.epoch
USING_TIME = args.using_time
START_EPOCH = args.resume_epoch
RESUME_FILE = args.resume_ckpt
VAR_DILATION = args.var_dilate
VAR_DILATION_VAL = args.var_dilate_val
SELF_COND = args.self_cond
MODEL_NAME = f"{notes}_epoch{EPOCH_NUM}_maxlen_{MAX_LENGTH}_x_0_predict{X_0_PREDICTION}__use_x_t{USE_X_T_LOSS}_use_x_1{USE_X_1_LOSS}_use_prob{USE_PROB_LOSS}"
RESULT_FILE = f'{MODEL_NAME}_res'
LOG_DIR = args.logdir
accelerator.print(f"trial name: {MODEL_NAME}")
