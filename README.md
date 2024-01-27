# Exploring Strategies for Utilizing Vision-Language Models in Subtitle Placement

The provided code pertains to the Master's Thesis titled _Exploring Strategies for Utilizing Vision-Language Models in 
Subtitle Placement_. 

Although we cannot share the copyrighted dataset, we offer configuration files for model training along with evaluation 
scripts to reproduce our results. 

## Updates 
- 02/03/2024: Initial release.

## Models and Training
The work is based on the following repositories and paper:

| Model Type               | Model (Name)  | Paper                                         | Repo                                                            |
|--------------------------|---------------|-----------------------------------------------|-----------------------------------------------------------------|
| Vision Model             | Pix2Pix       | [Paper](https://arxiv.org/pdf/1611.07004.pdf) | [Repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
| Vision-Language Models   | GLIP          | [Paper](https://arxiv.org/abs/2112.03857)     | [Repo](https://github.com/microsoft/GLIP/tree/main)             |
| Vision-Language Models   | SAN           | [Paper](https://arxiv.org/abs/2302.12242)     | [Repo](https://github.com/MendelXu/SAN/tree/main)               |
| Vision-Language Models   | DALL-E 2      | [Paper](http://arxiv.org/pdf/2204.06125.pdf)  | N/A                                                             |

For training details on the models we refer to the repositories of the respective models. 

You can find relevant configuration files in the subfolder ``\configs``. The files are structured for each model, 
except DALL-E 2 which offers no fine-tuning capabilities.

SAN and GLIP have been trained and tested on Google Colab, which is why we also provide training scripts in the 
directory ``\configs\{model}\training``.
Due to various python, pytorch and CUDA dependencies, various code snippets are highly vulnerable in terms of Google 
Colab version updates. 
This is why we try to fix the version of python itself and the used libraries. Nevertheless, if a script is not working 
anymore, please check original repositories for further details.

## Dataset
We also provide the script that was used to build our dataset, however we are not able to provide the original video and subtitle file, due to copyright restrictions.
You will be able to find instructions for gathering the subtitle files from video streams in relevant forums.

The script needs the following data to be organized as follows:

## Evaluation


