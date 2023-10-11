---

# Repository MOOD Challenge - team TUe_IMAGe 

This page contains the code to reproduce the submission of team TUe_IMAGe for the [MICCAI Medical Out-of-Distribution (MOOD) Challenge 2023](http://medicalood.dkfz.de/web/).

This repository is based on [github.com/MIC-DKFZ/mood](https://github.com/MIC-DKFZ/mood), provided by the organizes of the challenge. Our model is partly based on [github.com/marksgraham/ddpm-ood](https://github.com/marksgraham/ddpm-ood), which we adapted slightly and can be found in `docker_submission/scripts/ddpm_ood`.

_Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Please make sure that your usage of this code is in compliance with the code license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/MIC-DKFZ/basic_unet_example/blob/master/LICENSE)



### Requirements


Install python requirements:

```
pip install -r docker_submission/requirements.txt
```

To test the docker submission, you need to install [docker](https://www.docker.com/get-started) and [NVIDIA container Toolkit](https://github.com/NVIDIA/nvidia-docker).


The organizers suggest the following folder structure to work with the provided examples:

```
data/
--- brain/
------ brain_train/
------ toy/
------ toy_label/
--- abdom/
------ abdom_train/
------ toy/
------ toy_label/
```



## Training
To train our model for the MOOD submission, we ran the following. **Note** that we made adaptations to the [ddpm-ood](https://github.com/evihuijben/mood_challenge/tree/master/docker_submission/scripts/ddpm_ood) repository. We are currently working on generalizing this submodule by avoiding absolute paths. 

```
python train_ddpm_mood.py --parallel --simplex_noise 1 --batch_size 4 --eval_freq 10 --checkpoint_every 10 --is_grayscale 1 --n_epochs 60 --beta_schedule 'scaled_linear_beta'  --beta_start 0.001 --beta_end 0.015
```

...


## Inference
#### Step 1: Add model checkpoints
Make sure to put the model checkpoints in the directory docker_submission/scripts/checkpoints

#### Step 2: Build Docker and test the model

Build and test the docker by running the following:

```
python docker_submission/run_example.py -i /data/brain
```

With `-i` you should pass the data input folder (which has to contain a _toy_ and _toy_label_ directory).

#### Step 3: Test the Docker for submission

To check whether the submission complies with the challenge format, run the following:

```
python scripts_mood_evaluate/test_docker.py -d mood_example -i /data/ -t sample
```
With `-i` you should pass the base data input folder (which has to contain a folder _brain_ and _abdom_, and both folders have to contain a _toy_ and _toy_label_ directory).

with `-t` you can define the Challenge Task (either _sample_ or _pixel_)


