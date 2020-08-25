# Automated Classification of Rat Ultrasonic Vocalizations

---

### Introduction

Rats are highly social creatures that live in large groups with established hierarchies. Social interactions between individuals involves the use of odours, whisker touch and ultrasonic vocalizations (USVs). In a previous study (Rao et al, 2014), a large dataset of >70,000 USVs were identified and manually labelled into various calltypes. Of these, the trill, complex and flat calltypes form the vast majority (~80%) of the USVs that are vocalised during social interactions. 

This project aims to train a Convolutional Neural Networks (CNN) to classify USVs into these 3 predominant calltypes. For comparision, a CNN developed for mouse USVs was also used (DeepSqueak, Coffey et al, 2019). 

### Methods

#### Data Acquisition

Rats were placed on two elevated platforms separated by a gap. Spontaneous facial interactions that consist of extensive whisker-to-whisker & snout-to-snout touch events were recorded using low-and high-speed cameras under infra-red illumination. Ultrasonic vocalisations were recorded by specialized microphones and saved as .wav files (Rao et al, 2014).

#### Digital Signal Processing

.wav files were processed with a band pass filter (to remove low frequency noise) and compute power spectral density (to detect lack of signal) in each of the microphones. Spectrograms of a fixed duration from the onset of each USV (100 ms) were generated by fast Fourier transformation. The SciPy package was used for these steps. Spectrograms were manually checked for noise and saved into directories as required by the Keras flow_from_directory method. 

(Due to size constraints, only a few sample spectrograms are provided in this repository)

#### Convolutional Neural Networks 

The CNNs are implemented in Python on a Keras backend to train on an AMD GPU (using PlaidML, much thanks to Ricardo Di Sipio). The code can be used to run on a TensorFlow backend by simply commenting out the 'Activate Keras backend' line and replacing all "import keras..."statements with "import tensorflow.keras...".

#### Installation
