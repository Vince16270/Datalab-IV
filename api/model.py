import numpy as np
import tensorflow as tf
import cv2

def load_model():
    model_path = '/Users/vince/School - Datalab IV/Modellen/fully_runned_model_deeplab_200ep.h5'
    def iou(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
        return tf.reduce_mean((intersection + 1e-15) / (union + 1e-15), axis=0)
    
    def dice_loss(y_true, y_pred, smooth=1):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    model = tf.keras.models.load_model(model_path, custom_objects={'iou': iou, 'dice_loss': dice_loss})
    return model

def predict(model, segment):
    expected_size = (256, 256, 3)
    if segment.shape != expected_size:
        segment = cv2.resize(segment, (expected_size[1], expected_size[0]))
    
    segment = np.expand_dims(segment, axis=0)
    prediction = model.predict(segment)
    return prediction[0]