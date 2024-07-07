  <h2>🦄️ Measurement Guidance in Diffusion Models: Insight from Medical Image Synthesis (TPMAI 2024) </h2>

<div>
    <a href='https://scholar.google.com.hk/citations?user=InHF3ykAAAAJ&hl=zh-CN' target='_blank'>Yinmin Luo</a> <sup>*,#</sup> &nbsp;
    <a href='https://github.com/yangqy1110' target='_blank'>Qinyu Yang</a><sup>*</sup> &nbsp;
    <a href='https://github.com/yangqy1110/MGDM' target='_blank'>Yuheng Fan</a> &nbsp;
    <a href='https://scholar.google.com.hk/citations?user=AWI7KUsAAAAJ&hl=zh-CN' target='_blank'>Haikun Qi</a> &nbsp;
    <a href='https://menghanxia.github.io/' target='_blank'>Menghan Xia</a> &nbsp;
</div>
<div>
    <sup>*</sup>First Author, &nbsp; <sup>#</sup> Corresponding Author. &nbsp;
  
The work is mainly completed by [Qinyu Yang](https://github.com/yangqy1110) and [Yuheng Fan](https://github.com/yangqy1110/MGDM) under the careful guidance of [Yinmin Luo](https://scholar.google.com.hk/citations?user=InHF3ykAAAAJ&hl=zh-CN).
</div>

## Introduction
In this work, we first conducted an analysis on previous guidance as well as its contributions on further applications from the perspective of data distribution. To synthesize samples which can help downstream applications, we then introduce uncertainty guidance in each sampling step and design an uncertainty-guided diffusion models.

<p align="center">
  <img src="assets/images/pipeline.png">
</p>

Furthermore, we provide a theoretical guarantee for general gradient guidance in diffusion models, which would benefit future research on investigating other forms of measurement guidance for specific generative tasks.

## 🕹️ Code and Environment

#### 1. Clone the Repository

```
git clone https://github.com/yangqy1110/MGDM.git
cd ./MGDM/scripts
```

#### 2. Environment Setup

First configure the environment according to [guided-diffusion](https://github.com/openai/guided-diffusion) and [improved-diffusion](https://github.com/openai/improved-diffusion).
```
# Finally:
pip install blofile
conda install mpi4py
pip install torchsampler
```

## 💫 Step 1: Pre-training

### 1. Diffusion Model

```Python
(CUDA_VISIBLE_DEVICES=$device )python image_train.py --single_gpu True # specific single gpu(default is 0)
mpiexec -n $gpu_num python image_train.py                              # multi-gpu parallel
```

```
--data_dir              # Path to training data.
--schedule_sampler      # Default is "uniform".
--lr                    # learning rate, default is 1e-4.
--weight_decay          # Default is 0.00001.
--lr_anneal_steps       # Total training steps. The default value is False, which means unlimited.
--batch_size            # Default is 1.
--microbatch            # Default is -1, disables microbatches.
--ema_rate              # Default is "0.9999",  # comma-separated list of EMA values.
--log_interval          # How many steps to print a log? default is 10.
--save_interval         # How many steps to save a checkpoint? default is 10000.
--resume_checkpoint     # Initial model weight pth file path, default is "".
--use_fp16              # Default is False.
--fp16_scale_growth     # Default is 1e-3.
--log_root              # FolderPath of log and checkpoint "../logs".
--imablancedsample      # Whether to use torchsampler.ImbalancedDatasetSampler. The default value is False. 
                        # Not available when using multiple GPUs. Default is True.
--random_flip           # Whether to use np.fliplr.
--single_gpu            # Whether to specify a single GPU. The default value is False.
--image_size            # size of image. Default is 128.
--in_channels           # The number of channels entered into the network. Default is 3.
--num_classes           # Several classification questions. Default is 2.
--prob_uncon            # Probability of classless embedding in training. Default is 0.
```

Other hyperparameters can be found in `def diffusion_defaults()` and `def model_and_diffusion_defaults()` in `guided_diffusion/script_util.py`.
