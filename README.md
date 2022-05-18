# Deep Fake Audio Classifier
This is the submission for the Project 2 and 3 for the graduate level Deep Learning course (GY-7123) at NYU Tandon. <br>
<br>

**Developers: Abhishek Rathod, Jake Gus, Utkarsh Shekhar**    
**Course: ECE-GY 7123 Spring 2022**

## Overview
The objective of this project is create a classifier for differentiating between real audio and generated audio (created using 6 different architectures)

## Model Architechtures
3 models are available for direct usage : 
<ol>
  <li>The CNN model currently uses 4 convolution blocks (each containg a conv, ReLU and BatchNorm layer) with kaiming normal followed by average pooling and a linear classifier.</li>
  <li>The standard ResNet-18 with channel and parameter modifications</li>
  <li>A Temporal Convolutional Network</li>
</ol>


## Dataset
The dataset for the audio clips is available at  : 
<ol>
  <li>[LJSpeech](https://keithito.com/LJ-Speech-Dataset/).</li>
  <li>[WaveFake](https://zenodo.org/record/5642694#.YmYABNPMKre) [1]</li>
  <li>[JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) , Use this for Unseen Testing data</li>
</ol>

## Training the Data
<ol>
  <li>After Downloading the Datasets please replace the directory links in "data_script.py". This scripts helps sanitize and pre-process the audio data.</li>
  <li>Change the model that you are running in "audio_detect.py" </li>
  <li>Run the code using "audio_detect.py" </li>
</ol>

## Recreating the results
<ol>
  <li>Some pre-trained models for CNN and ResNet-18 are provided.</li>
  <li>Load them by changing the model and path in "inference.py" </li>
  <li>Modify the directory for data in "inference.py" for JSUT dataset and run the code </li>
</ol>

## References
<a id="1">[1]</a> 
Frank, J., & Schönherr, L. (2021). WaveFake: A Data Set to Facilitate Audio Deepfake Detection. arXiv preprint arXiv:2111.02813.
