# Combined CNN and RNN model for emotion recognition based on facial expressions from videos

This project is based on Denis Rangulov and Muhammad Fahim paper [Emotion Recognition on large video dataset based on 
Convolutional Feature Extractor and Recurrent Neural Network](https://arxiv.org/abs/2006.11168) and their repository
[Combined-CNN-RNN-for-emotion-recognition](https://github.com/DenisRang/Combined-CNN-RNN-for-emotion-recognition)

## Installation

### Python
Install Python 3.10. Instructions for installing Python on your system can be found [here](https://www.python.org/downloads/).

### Environment
Create and activate virtual environment:
 ```sh
python -m venv env
source env/bin/activate
 ```
or if you are using Conda:
 ```sh
conda create --name cnn-rnn-emotion-recognition python=3.8
conda activate cnn-rnn-emotion-recognition
 ```
### Dependencies
Install required libraries:
 ```sh
 pip install -r requirements.txt
 ```
or if you have ARM CPU like Apple M1:
 ```sh
 pip install -r requirements-arm.txt
 ```
