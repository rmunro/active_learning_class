# Active Learning Class
Code for class on Deep Active Learning and Annotation

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

* Assuming the pip installation of tensflow. See: https://www.tensorflow.org/install/



