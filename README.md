# Active Learning Class
Code for class on Deep Active Learning and Annotation

You can download the data of Sports Images from:  https://www.dropbox.com/s/j7kyqu04slc0oiz/data.tar.gz

(Warning, it's about 4GB). If you want to download it from the command line, use:

`curl -L -o data.tar.gz https://www.dropbox.com/s/j7kyqu04slc0oiz/data.tar.gz`

install your data at the same base location as the code. You should see these directories:

model_files: the DL model and the cache of the image-net layers for each image.

raw_data: the 'unlabeled' images (potentially already labeled for the sake of this class).

test_data: the randomly selected held-out test data with which to evaluate accuracy.

training_data: the data currently labeled and available for training / retraining the model.

## Usage*:

`python retrain.py` 

retrains the model with the data in the training_data directory

`python get_predictions.py` 

get the accuracy of the current model on the 'test_data' data.

`python get_predictions.py --directory=raw_data` 

get predictions on the 'unlabeled' data, and apply Active Learning strategies to order the raw images for human annotation

When running on the raw_data, you will get the ordered outputs from the different active learning strategies in the code, with the most important item to label first. This raw data has the actual (oracle) labels. Depending on how many labels you want to 'annotate' in each cycle, you can move that image from the raw_data directory to the corresponding training directory. In reality, of course, your raw, unlabeled images will not have their labels.

## Installing 

This assuming the pip installation of tensflow. See: https://www.tensorflow.org/install/

At a minimum, you *might* get away with installing only `git` `numpy` and `tensorflow` only:

`sudo yum install git`

`sudo -H pip install numpy` 

`sudo -H pip install tensorflow`

... but I recommend working out what's best in your environment, and be careful with 'sudo's that you meet on the internet!






