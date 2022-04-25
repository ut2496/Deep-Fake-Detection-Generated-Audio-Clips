# Deep Fake Audio Classifier
This is the submission for the Project 2 and 3 for the graduate level Deep Learning course (GY-7123) at NYU Tandon. <br>
<br>
**This is currently a WIP**


**Developers: Abhishek Rathod, Jake Gus, Utkarsh Shekhar**    
**Course: ECE-GY 7123 Spring 2022**

## Overview
The objective of this project is create a classifier for differentiating between real audio and generated audio (created using 6 different architectures)

## Model Architechture and implementation
The model currently uses 4 convolution blocks (each containg a conv, ReLU and BatchNorm layer) followed by average pooling and a linear classifier.

## Dataset
The dataset for the audio clips is available at  : 
<ol>
  <li>[LJSpeech](https://keithito.com/LJ-Speech-Dataset/).</li>
  <li>[WaveFake](https://zenodo.org/record/5642694#.YmYABNPMKre) [1]</li>
</ol>

## Recreating the results
<ol>
  <li>After Downloading the Datasets please place it in the directory as specified in "data_script.py" . This scripts helps sanitize and pre-process the audio data.</li>
  <li>Run the code using "audio_detect.py" </li>
</ol>

## References
<a id="1">[1]</a> 
Frank, J., & Schönherr, L. (2021). WaveFake: A Data Set to Facilitate Audio Deepfake Detection. arXiv preprint arXiv:2111.02813.
