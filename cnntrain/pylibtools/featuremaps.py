from tensorflow.python.util import tf_decorator


import tensorflow as tf


def featuremap(model):
    for i in range(len(model.layers)):
        print(model.layers[i])
    layer_outputs = [layer.output for layer in model.layers]
    feature_map_model = tf.keras.models.Model(input=model.input, output=layer_outputs)