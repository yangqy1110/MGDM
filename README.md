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
