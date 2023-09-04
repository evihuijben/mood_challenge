---

# Repository MOOD Challenge - team TUe_IMAGe 

This page contains the code to reproduce the submission of team TUe_IMAGe for the [MICCAI Medical Out-of-Distribution (MOOD) Challenge 2023](http://medicalood.dkfz.de/web/).

This repository is based on [github.com/MIC-DKFZ/mood](https://github.com/MIC-DKFZ/mood), provided by the organizes of the challenge.

_Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Please make sure that your usage of this code is in compliance with the code license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/MIC-DKFZ/basic_unet_example/blob/master/LICENSE)



### Requirements


Install python requirements:

```
pip install -r requirements.txt
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


...


## Inference
#### Step 1: Add model checkpoints
Make sure to put the model checkpoints in the directory docker_submission/scripts/checkpoints

#### Step 2: Build Docker and test the model

Build and test the docker by running the following:

```
python docker_example/run_example.py -i /data/brain
```

With `-i` you should pass the data input folder (which has to contain a _toy_ and _toy_label_ directory).

#### Step 3: Test the Docker for submission

To check whether the submission complies with the challenge format, run the following:

```
python scripts_mood_evaluate/test_docker.py -d mood_example -i /data/ -t sample
```
With `-i` you should pass the base data input folder (which has to contain a folder _brain_ and _abdom_, and both folders have to contain a _toy_ and _toy_label_ directory).

with `-t` you can define the Challenge Task (either _sample_ or _pixel_)


