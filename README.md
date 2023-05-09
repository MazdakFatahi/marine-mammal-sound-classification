## Dataset: 

We used 10 classes of the Watkins Marine Mammal Sound Database

https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm
## Dependencies:

- pycochleagram (https://github.com/mcdermottLab/pycochleagram)
- scipy
- librosa
- torch
- torchvision
- pandas

## helper package

- `signal_utils.py`:
Contains functions for generating the required graphs for representing the signals (wave files) in time-frequency domain:

    - melspectrogram_db
    - spectogram (using `pyplot package` or `librosa package`)
    - cochleagram
- `train_utils.py`:
The code contains functions for:

- splitting the data into test and train
- preparing the data classes (putting the graphs in the corresponding folders) according to the label csv file.
- train and test functions
  - Note: here we used vgg16 pre-trained network. You can define your model and pass it into the train and test


## Example 1:

Please refer to the `extract_train_pipeline.ipynb` to see how you can use everything in a pipeline.

## Example 2:

To compare the results of using different type of the graphs or the same method with different parameters, you can prepare different datasets and train them. `extract_features_test_train.ipynb` shows how to define different parameters for generating different datasets using the Mel-Spectogram. Also, here we extracted the features using the pre-trained vgg16 (for using in training phase with a much more simpler network). So, here for the mel-Spectogram, the number of the FFT and for extracting the features, the batch-size effect are considered. 

## Example 3:

`multiple_run_using_prepared_classes.ipynb` shows how to use the generated datasets for training the same network and storing the results for exploring the effect of changing the parameters.