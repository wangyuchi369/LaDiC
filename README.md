
# Quick Start of LaDiC(version for review)
## Environment
Required packages and dependencies are listed in the `ladic.yaml` file. You can install the environment using Conda with the following command:
```bash
conda env create -f ladic.yaml
```

## Datasets
Download [MSCOCO dataset](https://cocodataset.org/#download) and place it into `datasets` folder.

Meanwhile, we follow Karpathy split, and its annotation files can be found in its [orginial paper.](https://cs.stanford.edu/people/karpathy/deepimagesent/)

## Required pretrained models
In our LaDiC model, Text Encoder and Decoder are initialized from BERT-Base-uncased, which can be downloaded from [Huggingface](https://huggingface.co/bert-base-uncased).

As for image encoder, we utilized pretrained ViT in BLIP.You may download from [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth) and put it into `pretrained_ckpt` folder. More information can be found in [BLIP's official repo](https://github.com/salesforce/BLIP)


## Accelerate Configuration
We use accelerate package developed by Huggingface.

Configure Accelerate by using the following command in the command line:
```bash
accelerate config
```
Answer the questions based on your actual setup. You will be prompted to specify the GPU to use, and other configurations can be left as default. For more information, refer to [this link](https://huggingface.co/docs/accelerate/v0.13.2/en/quicktour#launching-your-distributed-script).

## Training
Launch the `main.py` script using Accelerate with the following command:
```bash
accelerate launch main.py [--args]
```

We list some important optional parameters as follows. The `notes` parameter is both a note to be placed at the top of the filename and the running name for Wandb. More hyperparameters and their description can be found in `configs/`
```bash
parser.add_argument('--notes', type=str, default=None, help='Note to be included in the trial name')
parser.add_argument('--bsz', type=int, default=5, help='batch size')
parser.add_argument('--seqlen', type=int, default=80, help='sequence length')
parser.add_argument('--epoch', type=int, default=10, help='epoch num')
parser.add_argument('--resume_epoch', type=int, default=0, help='start epoch of resume')
parser.add_argument('--resume_ckpt', type=str, default=None, help='resume or not')
parser.add_argument('--logdir', type=str, default='checkpoint', help='logdir')
```



## Evaluation

Specify `MODEL_NAME` and `RESULLT_FILE` in `coco_eval.py` representing checkpoint to be evaluated and output path respectively. Then you can run 
```bash
python coco_eval.py
```

### *Best wishes!*

