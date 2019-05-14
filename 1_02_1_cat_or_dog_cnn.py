#%%
from __future__ import absolute_import, division, print_function

#%%
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

#%%
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

#%%

# load json and create model
from keras.models import model_from_json

with tf.Graph().as_default():
    with tf.Session() as sess:
        json_file = open('model/cat_or_dog.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights("model/cat_or_dog.h5")
        print("Loaded model from disk")
        model.compile(optimizer='adam', 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.save(sess, './model/cat_or_dog.ckpt')
        tf.train.write_graph(sess.graph.as_graph_def(), '.', 'cat_or_dog.pbtxt', as_text=True)

#%%
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph('./model/cat_or_dog.pbtxt', "", False,
                        "./model/cat_or_dog.ckpt", "dense_2/Sigmoid",
                        "save/restore_all", "save/Const:0",
                        "./model/frozen_cat_or_dog.pb", True, "")

#%%
from tensorflow.python.tools import optimize_for_inference_lib
inputGraph = tf.GraphDef()
with tf.gfile.Open('./model/frozen_cat_or_dog.pb', "rb") as f:
    data2read = f.read()
    inputGraph.ParseFromString(data2read)

outputGraph = optimize_for_inference_lib.optimize_for_inference(
                inputGraph,
                ["conv2d_1_input"],
                ["dense_2/Sigmoid"],
                tf.int32.as_datatype_enum)
f = tf.gfile.FastGFile('./model/opt_cat_or_dog.pb', "w")
f.write(outputGraph.SerializeToString())

#%%
gf = tf.GraphDef()
gf.ParseFromString(open('./model/frozen_cat_or_dog.pb','rb').read()) 
[n.name + '=>' +  n.op for n in gf.node if n.op in ( 'Softmax','Placeholder')]      
#%%
from keras.preprocessing import image
input_size = (128, 128)

test_image = image.load_img('/Users/sheik/repo/learning/ai/dla2z/CNN/dataset/single_prediction/cat_or_dog_2.jpg', target_size = input_size)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
prediction = 'Not yet'

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)     


#%%
def save_invalid_pb(model, filename):
    # First freeze the graph and remove training nodes.    

    output_names = model.output.op.name
    model_weights = model.get_weights()
    sess = tf.keras.backend.get_session()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    model.set_weights(model_weights)

    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    # Save the model
    with open(filename, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())

#%%
#save(model, filename="model/cat_or_dog.pb")