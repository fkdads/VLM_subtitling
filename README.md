# Exploring Strategies for Utilizing Vision-Language Models in Subtitle Placement

The provided code pertains to the Master's Thesis titled _Exploring Strategies for Utilizing Vision-Language Models in 
Subtitle Placement_. 

Although we cannot share the copyrighted dataset, we offer configuration files for model training along with evaluation 
scripts to reproduce our results. 

## Updates 
- 02/03/2024: Initial release.

## Tasks and Experiments {#experiments}
| Task ID | Task Name | Task Description | Experiments                                                                                                         |
|---------|-----------|------------------|---------------------------------------------------------------------------------------------------------------------|
| 1 | Subtitle Placement Based on Original Image Input | Predict Subtitle Placement directly from the video frames.| - Single Frame without Default Subtitle Placement Marker<br/> - Single Frame with Default Subtitle Placement Marker |
| 2 | Active Speaker Detection Based on Original Inmage Input | Predict location of Active Speaker from native video frames. | - Single Frame <br/>- Three Frames Overlapping<br/>- Three Frames Voting                                            |
| 3 | Person Detection Based on Original Image Input | Highlight all persons visible in natvie video frame. | - Single Frame                                                                                                      |
| 4 | Active Speaker Detection Based on Person Detection Output | Detect Active Speaker in frames highlighting visible persons. | - Single Frame                                                                                                      |
| 5 | Subtitlte Placement Based on Active Speaker Detection | Predict subtitle placement based on output of Task 2 and Task 4 | - Best working setup of respective Task                                                                             |


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
We also provide the script that was used to build our dataset, however we are not able to provide the original video 
and subtitle file, due to copyright restrictions.
You will be able to find instructions for gathering the subtitle files from video streams in relevant forums.

### Source Data
The script needs the following data to be organized as follows:
- Provide the XML-based subtitle file in ``\dataset\subtitles\subtitle.xml`` (you can also provide it somewhere else 
and set path with input parameter --stp)
- Provide video file in ``\dataset\video\video.mp4`` (you can also store the video in any other directory and provide 
its location via --vp input parameter)

### Build Dataset
We have established several tasks and sub-experiments, as listed in Section [Tasks and Experiments]{#experiments}. 
The relevant steps are summarized in the linked publication. 

We will distinguish between building the following datasets:

| Dataset ID | Dataset Description | Related Tasks and Experiments |
|------------|---------------------|-------------------------------|
| 1          | Single Frame | Dataset focusing on single frames provided as input. Input frames consists of frames with default subtitle placement marker and without a default subtitle placement marker | All single frame experiments |
| 2          | Three Frames Overlapped | Dataset consisting of input frames that stack three consecutive frames into a single frame | All three frames overlapped experiments |
| 3          | Three Frames Voting | Dataset sampling three individual, consecutive frames, that will be evaluated to determine majority consensus. While the frames are stacked in Dataset 2, they provide three dinstinctive samples files. | All three frames voting experiments |

To build the dataset, you can use the following command lines:
- **Dataset - Single Frame**: ``--vo
7
--box
--nf
2500
--tkn
subtitle_position_boxes_middle_of_subtitle
--fsp
1650
--mos
--emd``
- **Dataset - Three Frames Overlapped**: ``--vo
7
--box
--nf
2500
--tkn
subtitle_position_boxes_middle_of_subtitle_overlapped
--fsp
1650
--mos
--emd
--overlay_frames
3
--overlay_frames_skip
1``
- **Dataset - Three Frames Voting**: ``--vo
7
--box
--nf
2500
--tkn
subtitle_position_boxes_middle_of_subtitle_voting
--fsp
1650
--mos
--emd
--fps
3``

In order to generate the pixelmaps for SAN, you need to run the following script for each dataset again. But please
take care of installing [Segment Anything](https://github.com/facebookresearch/segment-anything/tree/main) first and
download the default/ViT-H model to your execution runtime.
- ``--vp 'D:\subtitle_placement_data_single\data\_A' --anno "D:\subtitle_placement_data_single\jsons\train\result.json", "D:\subtitle_placement_data_single\jsons\val\result.json", "D:\subtitle_placement_data_single\jsons\result.json"``
- You need to adjust the pathes and run the script individually for single, overlapped and voting experiment pre-processing.



> **Note:** You need to adjust the parameters --vo and --fsp before applying to you video file. The --vo (video offset) 
> parameter offers the possibilty to provide the offset between video recording start and subtitle file reference start 
> point in seconds. The --fsp (fixed_start_point) gives the flexibility to define the start point of the video sampling
> to skip introduction scenes.


To get insights on the available input parameters use the default help command in combination with the 
*dataset_sampler.py* file.

## Evaluation


