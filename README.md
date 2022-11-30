# entrainmentInfrastructure

<p> <a href="https://aws.amazon.com" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="aws" width="90" height="40"/> </a> <a href="https://www.gnu.org/software/bash/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/gnu_bash/gnu_bash-icon.svg" alt="bash" width="40" height="40"/> </a> <a href="https://www.docker.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> </a> <a href="https://golang.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/go/go-original.svg" alt="go" width="40" height="40"/> </a> <a href="https://www.linux.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> <a href="https://www.mathworks.com/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Matlab_Logo.png" alt="matlab" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://www.qt.io/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/0b/Qt_logo_2016.svg" alt="qt" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> <a href="https://unity.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/unity3d/unity3d-icon.svg" alt="unity" width="40" height="40"/> </a>  <a href="https://www.terraform.io/" target="_blank" rel="noreferrer"> <img src="https://img.shields.io/badge/terraform-%235835CC.svg?style=for-the-badge&logo=terraform&logoColor=white" alt="unity" width="90" height="40"/> </a></p>

## Description
This repository consists of several aspects of code needed to run an entrainment experiment using AWS and an Oculus Quest 2.

The repository is divided into different directories:
- lambda_functions: Lambda functions deployed to AWS that allow the experiment to be run. Deployment is handled by Terraform and Docker.
- data: Encloses Data analysis and meta analysis methods used to clean and analyse data recorded using a power and connectivity analysis. participant_info.py files are required to specify participant information of recorded data.
  - Further, holds data recorded using the custom suite developed.
  - Holds code to plot figures from meta analysis methods run on data.
  - Holds results of data analysis once run on recorded data.
- testing_interface: Holds code used for testing interface of experiment. This encompasses code to record data from a Gtec HIAmp using the API, Machine Learning code to use Q learning to lean an optimal policy to control the entrainment stimulus and all other aspects required to run the experiment which includes GUI and interfaces used to run the experiment.
- runners: Code to wrap docker and terraform commands required to run and deploy the experiment.
- tst: Code used to test various aspects of the system including the AWS infrastructure.
- Dockerfile: Docker file for experiment

## Using the Runner
Runner's allow easier actions for terraform and docker. From the runner's directory run:

| Description | Command|
| --- | ---|
| To deploy terraform changes (will create a docker image if needed) | `python runner.py --f tf --a apply`|
| To pull down infrastructure | `python runner.py --f tf --a destroy` |
| Further terraform based commands are possible: `validate, fmt, apply, plan, state, destroy, init` using the following | `python runner.py --f tf --a command`|
| Create the Docker image. Images can be `experiment_setup_only` (setting up the experiment Terraform) or , `experiment_complete` (running the experiment) | `python runner.py --f docker --a build -t targeted image`|
| Delete a Docker image | `python runner.py --f docker --a delete -t targeted image`|

## General workflows:
![](Images/IAC_entrainment.jpg?raw=true)
![](Images/Architecture.jpg?raw=true)
- tst directory has an `eeg_device_testing.py` file that is useful to record data from the gtec device without using a GUI.
- Infrusture has been designed to work with the correpsonding game developed in Unity for the Oculus Quest 2. The code for this should be in the same directory as this project and can be found at [OculusEntrainment](https://github.com/RC-7/OculusEntrainment).

Note: Auth lambda emails participants who have been authenticated to identify themselves. The Email password will be stored in AWS Secrets Manager. This step needs to be completed via the console. Currently, it is an env variable that needs to be manually set for the Lambda. This has been done for the privacy of AWS Secrets Manager.

TODO:
- Rework the participant_info system to hold information around participant data recorded.
- Improve explanation of different aspects of the code base.
- Clean code.
- Update `pythonPackages.txt` file