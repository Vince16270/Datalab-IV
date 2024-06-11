import numpy as np
import tensorflow as tf
import cv2

def load_model():
    model_path = '/Users/vince/School - Datalab IV/model_unet_300_ep.h5'
    def iou(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
        return tf.reduce_mean((intersection + 1e-15) / (union + 1e-15), axis=0)
    
    model = tf.keras.models.load_model(model_path, custom_objects={'iou': iou})
    return model

def predict(model, segment):
    expected_size = (256, 256, 3)
    if segment.shape != expected_size:
        segment = cv2.resize(segment, (expected_size[1], expected_size[0]))
    
    segment = np.expand_dims(segment, axis=0)
    prediction = model.predict(segment)
    return prediction[0]