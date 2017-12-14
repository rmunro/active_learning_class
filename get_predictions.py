# Copyright 2017 CrowdFlower, Inc.
#
# The Code below is adapted from Tensorflow code that is open source and available under Apache License at:
# https://github.com/tensorflow
# The TensorFlow Authors. All Rights Reserved.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Code to get predictions across a number of labels and order those labels for active learning

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import re
import operator
import glob
import random
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument(
    '--directory',
    type=str,
    default='test_data',
    help='Evaluation data.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=5,
    help='Display this many predictions.')
parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='DecodeJpeg/contents:0',
    help='Name of the input operation')
parser.add_argument(
    '--graph',
    default='model_files/output_graph.pb',
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    default = 'model_files/output_labels.txt',
    type=str,
    help='Absolute path to labels file (.txt)')


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(sess, image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  # Feed the image_data as input to the graph.
  #   predictions will contain a two-dimensional array, where one
  #   dimension represents the input image count, and the other has
  #   predictions per class
  softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)

  predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

  # Sort to show labels in order of confidence
  prediction_index = num_top_predictions
  top_k = predictions.argsort()[-num_top_predictions:][::-1]

  label_predictions = {}
    
  for node_id in top_k:
    human_string = labels[node_id]
    score = predictions[node_id]
    #print('%s (score = %.5f)' % (human_string, score))
    label_predictions[human_string] = score
      
  return label_predictions



def get_image_prediction(sess, image, labels):
  """Get a prediction for a given image with the given labels."""
  if not (image.endswith('.jpg') or image.endswith('.jpeg')):
    return []
  
  if not tf.gfile.Exists(image):
    tf.logging.fatal('image file does not exist %s', image)

  image_data = load_image(image)

  predictions = run_graph(sess, image_data, labels, FLAGS.input_layer, FLAGS.output_layer,
                          FLAGS.num_top_predictions)

  return predictions



def cp_file(path, directory):
  """Copies the given file to the new directory"""
  
  if not os.path.exists(directory):
    os.makedirs(directory)

  label = re.sub(FLAGS.directory+"\/","",path)
  label = re.sub("[^\/]*$","",label)  
  sub_directory = directory+"/"+label
  filen = re.sub(".*\/","",path)  
  
  if not os.path.exists(sub_directory):
    os.makedirs(sub_directory)

  target = sub_directory+filen

  if os.path.isfile(target):
    return False
  else:
    os.system("cp "+path+" "+target)
    return True


  
def main(argv):
  """Runs inference on images."""
  if argv[1:]:
    raise ValueError('Unused Command Line Args: %s' % argv[1:])

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)
    
  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)
                          
  # load labels
  labels = load_labels(FLAGS.labels)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)

  #look in directory
  label_dirs = os.listdir(FLAGS.directory)
  label_dirs_path = [FLAGS.directory+"/" + x for x in label_dirs] 

  microfscores = 0.0 # sum of f scores * counts
  macrofscores = 0.0 # sum of f scores

  total_images = 0
  total_labels = 0

  all_predictions = [] # record of all predictions, to use for active learning

  # With current tensorflow session, get predictions
  with tf.Session() as sess, tf.Graph().as_default():
    
    # for each subdirectory, corresponding to one label
    # NB: could remove this inner loop for active learning on raw data where the labels are not known
    for label_dir in label_dirs_path:
      total_labels += 1
      label = re.sub("^.*\/", '', label_dir)
      label = re.sub("[^a-z0-9]+", ' ', label.lower()) # get label name from dir, removing forbidden characters      

      files = os.listdir(label_dir)
      images = [label_dir+"/" + x for x in files]    

      count = 0 # number of images seen overall
      tp = 0 # number of true positives overall
      fn = 0 # number of false negatives overall
      fp = 0 # number of false positives overall

      for image in images:
        # if random.random() > 0.001:  # speed up test by reducing to just 3%
        #  continue
        
        image = image.rstrip()
        
        # Get all predictions for this image!!
        predictions = get_image_prediction(sess, image, labels)

        # Get the two most confidently predicted labels
        top_prediction = ""
        top_confidence = 0.0
        second_prediction = ""
        second_confidence = 0.0
        for prediction in predictions:
          confidence = predictions[prediction]
          if confidence > top_confidence:
            if top_confidence > second_confidence:
              second_prediction = top_prediction 
              second_confidence = top_confidence
            top_prediction = prediction
            top_confidence = confidence
          elif confidence > second_confidence:
            second_prediction = prediction
            second_confidence = confidence

        if top_confidence == 0.0 or second_confidence == 0.0:
          continue # something went wrong, most likely a corrupted file
        
        # get the ratio of top confidence to second most confident
        ratio = top_confidence / second_confidence
        rand = random.random()
        difference = top_confidence - second_confidence
        
        # WHAT WE WANT TO CAPTURE ABOUT EACH IMAGE TO USE FOR ACTIVE LEARNING
        image_info = [image, top_prediction, top_confidence, second_prediction, second_confidence, ratio, rand, difference]
        # print(image_info)
        all_predictions.append(image_info)

        # update accuracy metrics
        if top_prediction == label :
            tp+=1
        else:
            fn+=1
            fp+=1

        count += 1
        total_images +=1
        running_accuracy = tp / count

      # for this label, report accuracy
      if tp == 0:
        fscore = 0.0
      else:
        precision = tp / (tp + fp)
        recall = tp / ( tp + fn )
        fscore = (2* precision * recall) / (precision + recall)
      # print(label+ " f-score:")
      # print(fscore)
      
      microfscores += fscore * count 
      macrofscores += fscore

  # report overal accuracies
  microf = microfscores / total_images
  macrof = macrofscores / total_labels
  print("Micro-f: ")
  print(microf)
  print("Macro-f: ")
  print(macrof)


  if FLAGS.directory == "test_data":
    exit()
  
  # IMPLEMENT STRATEGY FOR ACTIVE LEARNING
  
  
  print("CONFIDENCE")
  # 1. ORDER BY THE LEAST CONFIDENT TO MOST CONFIDENT
  all_predictions.sort(key=lambda x: x[2], reverse=False)
  c = 0 
  for image_info in all_predictions:
    print(image_info[0])
    if c < 2000:
      if cp_file(image_info[0], "training_data_confidence"):
        c+=1


        
  print("RATIOS")    
  # 2. ORDER BY THE CLOSEST RATIOS
  all_predictions.sort(key=lambda x: x[5], reverse=False)
  c = 0
  for image_info in all_predictions:
    print(image_info[0])
    if c < 2000:
      if cp_file(image_info[0], "training_data_ratios"):
        c+=1
    
  
  # 3. STRATIFY BY LABEL, ENSURING EQUAL DISTRIBUTION ACROSS PREDICTED LABELS
         # Could be used in combination with 1. or 2.

  print("STRATIFIED")
  all_predictions.sort(key=lambda x: x[6], reverse=False) #randomize
  ordered_labels = {} # dict of list for each label.
  for image_info in all_predictions:
    top_prediction = image_info[1]
    if not top_prediction in ordered_labels:
      ordered_labels[top_prediction] = []
    ordered_labels[top_prediction].append(image_info[0]) # add url to list for that label

  print(ordered_labels)
  # exit()
    
  # interleave the per-label lists
  keep_going = True # to track whether there are any remaining to be ordered
  c = 0
  while keep_going:
    keep_going = False
    for label in ordered_labels:
      images = ordered_labels[label]
      print("There are "+str(len(images))+" images in "+label)
      if len(images) > 0:
        image = images.pop(0)
        keep_going = True
        print(image)
        if c < 2000:
          if cp_file(image, "training_data_stratified"):
            c+=1

  
  print("PAIRS")
  # 4. STRATIFY BY LABELS, ENSURING EQUAL DISTRIBUTION ACROSS PAIR OF PREDICTED LABELS
          # Could be used in combination with 1. or 2.
  all_predictions.sort(key=lambda x: x[5], reverse=False) 
  ordered_label_pairs = {} # dict of lists for each pair of labels label.
  for image_info in all_predictions:
    top_prediction_pair = image_info[1]+" "+image_info[3]
    if not top_prediction_pair in ordered_label_pairs:
      ordered_label_pairs[top_prediction_pair] = []
    ordered_label_pairs[top_prediction_pair].append(image_info[0]) # add url to list for that label pair

  # interleave the per-label-pair lists
  keep_going = True # to track whether there are any remaining to be ordered
  c = 0
  while keep_going:
    keep_going = False
    for label in ordered_label_pairs:
      images = ordered_label_pairs[label]
      if len(images) > 0:
        image = images.pop(0)
        keep_going = True
        print(image)
        if c < 2000:
          if cp_file(image, "training_data_pairs"):
            c += 1

            
  print("DIFFERENCE")
  # 5. ORDER BY THE DIFFERENCE BETWEEN MOST AND LEAST CONFIDENCE
  all_predictions.sort(key=lambda x: x[7], reverse=False)
  c = 0 
  for image_info in all_predictions:
    print(image_info[0])
    if c < 2000:
      cp_file(image_info[0], "training_data_difference")
    c += 1

    
  print("RANDOM")
  # 6. RANDOM ORDER FOR A BASELINE
  all_predictions.sort(key=lambda x: x[7], reverse=False)
  c = 0
  for image_info in all_predictions:
    print(image_info[0])
    if c < 2000:
      if cp_file(image_info[0], "training_data_random"):
        c += 1


  print("PATHOLOGICAL")
  # 7. ORDER BY THE MOST CONFIDENT TO LEAST CONFIDENT
  all_predictions.sort(key=lambda x: x[2], reverse=TRUE)
  c = 0
  for image_info in all_predictions:
    print(image_info[0])
    if c < 2000:
      if cp_file(image_info[0], "training_data_pathological"):
        c+=1



  print("PATHSTRAT")
  # 8. ORDER BY THE MOST CONFIDENT TO LEAST CONFIDENT, STRATIFIED 
  all_predictions.sort(key=lambda x: x[2], reverse=TRUE)
  ordered_labels = {} # dict of list for each label.
  for image_info in all_predictions:
    top_prediction = image_info[1]
    if not top_prediction in ordered_labels:
      ordered_labels[top_prediction] = []
    ordered_labels[top_prediction].append(image_info[0]) # add url to list for that label
      
  # interleave the per-label lists
  keep_going = True # to track whether there are any remaining to be ordered
  c = 0
  while keep_going:
    keep_going = False
    for label in ordered_labels:
      images = ordered_labels[label]
      print("There are "+str(len(images))+" images in "+label)
      if len(images) > 0:
        image = images.pop(0)
        keep_going = True
        print(image)
        if c < 2000:
          if cp_file(image, "training_data_path_strat"):
            c+=1
            
            
            
        

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)



  
