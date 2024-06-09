import torch

from configs.config import *
from einops import repeat

if COSIN_SCHEDULE:
    def scheduler(t):
        s = 0.008  # small value prevent beta_t too small, from Improved DDPM paper
        return torch.cos(math.pi / 2 * (t / STEP_TOT + s) / (1 + s)) ** 2


    ts = torch.arange(STEP_TOT).to(accelerator.device)
    alpha_cumprod = scheduler(ts) / scheduler(torch.zeros(1, device=accelerator.device))
    prev = alpha_cumprod.roll(1, 0)
    prev[0] = 1
    alphas = alpha_cumprod / prev
    betas = 1 - alphas
else:
    betas = torch.hstack([torch.zeros(1), torch.linspace(BETA_MIN, BETA_MAX, STEP_TOT)]).to(accelerator.device)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas[:-1], 0)


def diffuse_t(x, t, is_test=False):
    '''
  input:
    x_shape: [batch_size, seq_len, IN_CHANNEL]
    t shape: [sample num,1,1]
      NOTE: not necessary have hyperparameter sample_size number of element, to allow single diffuse generation

  return shape [sample_num * batch_size, seq_len, IN_CHANNEL] 
  '''
    batch_size, seq_len, dim = x.shape
    # sample_shape = (t.numel(), *(1,) * len(x.shape))

    epsilon = torch.normal(0, 1, x.shape).to(accelerator.device)
    mean = torch.sqrt(repeat(alpha_cumprod[t], 'b -> b seq d', seq=seq_len, d=dim)) * x  # (bsz, seqlen, emb)
    if not is_test:
      bias = epsilon * torch.sqrt(repeat(1 - alpha_cumprod[t], 'b -> b seq d', seq=seq_len, d=dim)) * VAR_DILATION
    else:
      bias = epsilon * torch.sqrt(repeat(1 - alpha_cumprod[t], 'b -> b seq d', seq=seq_len, d=dim)) * VAR_DILATION_VAL
    return mean + bias


def generate_diffuse_pair(x_0, t, t_next=None):
    '''
  input:
    x_0 shape: [batch_size, seq_len, IN_CHANNEL],
    t shape: [sample_num]
      NOTE: not necessary have hyperparameter sample_size number of element, to allow single diffuse generation

  return (net input, net target)
    net input shape: [sample_num * batch_size, seq_len, IN_CHANNEL]
    net target shape: if t_next is None then [batch_size, seq_len, IN_CHANNEL] else [sample_num * batch_size, seq_len, IN_CHANNEL]
  '''
    if X_0_PREDICTION:
        # predict x_0
        return (diffuse_t(x_0, t), x_0)

    # predict x_{t_next}
    return diffuse_t(x_0, t), diffuse_t(x_0, t_next)


def p_sample(xt, x_0, t, delta):
    """
    calculate p(x_{t-\delta t} | x_t, x_0)
    """
    # xt_coeff = torch.sqrt(alphas[t]) * (1 - alpha_cumprod[t - delta])/ (1 - alpha_cumprod[t])
    # x0_coeff = torch.sqrt(alpha_cumprod[t - delta]) * betas[t] / (1 - alpha_cumprod[t])
    # mu = xt_coeff * xt + x0_coeff * x_0
    # sigma = (1 - alpha_cumprod[t - delta]) / (1 - alpha_cumprod[t]) * betas[t] * 0
    eta = 0.1
    noise = torch.normal(0, 1, xt.shape).to(accelerator.device)
    sigma = eta * (1 - alpha_cumprod[t - delta]) / (1 - alpha_cumprod[t]) * betas[t]
    mu = torch.sqrt(alpha_cumprod[t - delta]) * x_0 + torch.sqrt(1 - alpha_cumprod[t - delta] - sigma) \
         * (xt - torch.sqrt(alpha_cumprod[t]) * x_0) / torch.sqrt(1 - alpha_cumprod[t])
    return mu + torch.sqrt(sigma) * noise
