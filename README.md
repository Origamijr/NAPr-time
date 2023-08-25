Dropped. Successor project (RASPPPy)[https://github.com/Origamijr/RASPPPy]

# NAPr-time
Repository for Neural Audio Processing in real-time (hopefully). At the moment, contains the code for my [previous repository](https://github.com/Origamijr/Audio-Classify). I am in the process of rewriting the code.

## Installation
This project was implemented using pytorch.
```
pip install -r requirements.txt
```

There may be issues with installation order. Check directions in requirements just in case

## Preprocessing
Experiments were performed on the vctk dataset, classifying each audio clip by the speaker. the dataset can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3443). The raw data should be stored in a location specified by the "source" tag in config.toml.

To preprocess the data into a dataframe and store the preprocessed data into a HDF file, run the following command:
```
python preprocessing.py
```


## Train
Training was performed via notebooks on Google colab, but can also be performed via command line via the following command:
```
python train.py
```
I wish I had a cuda enabled GPU to train locally...
