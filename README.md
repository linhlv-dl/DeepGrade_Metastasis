# Deep Attention-MIL to predict the metastasis risk for Sarcoma patients



## Description

This project contains the source program of the Deep Attention Multiple Instance Learning (Deep Attention-MIL) model.
This model is used to provide the risk score of metastasis for the Sarcoma patients.
The study is based on the extracted features from different areas of tumors and these combinations.
The outputs of the model are used as an input in another model to analyze the role of each area in the treatment of Sarcoma cancer.
![description](docs/model.png)

## Installation

- Linux (Ubuntu 20.04.6 LTS)
- NVIDIA GPU (Nvidia Quadro RTX 6000)
- Python3 (3.8.10), PyTorch (2.2.2), torchvision (0.13.1), pytorch-lightning (1.5.7), numpy (1.21.5)


## The source code
The program to train/test the model are implemented in the following files:
<ul>
	<li><i>main_cv.py</i>: The main file to run the cross-validation test for classification task</li>
	<li><i>model_pl.py</i>: The classification model</li>
	<li><i>bergonie_dataloader_survival_wsi.py</i>: The file to load and create the Dataset, Dataloader</li>
	<li><i>inference_CV.py</i>: The inference process for classification cross-validation models</li>
</ul>
