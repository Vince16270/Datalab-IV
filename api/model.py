# Description: This file contains the code to load a pre-trained DeepLab model for image segmentation and make predictions using the model.

# Import the required libraries 
import numpy as np
import tensorflow as tf
import cv2

def load_model():
    """
    Loads a pre-trained DeepLab model for image segmentation.

    The model is loaded from the specified file path and custom loss functions
    and metrics are defined to be used during model loading.

    Returns:
        model (tf.keras.Model): The loaded Keras model.
    """
    model_path = '/Users/vince/School - Datalab IV/Modellen/fully_runned_model_deeplab_200ep.h5'

    def iou(y_true, y_pred):
        """
        Computes the Intersection over Union (IoU) metric.

        Args:
            y_true (tensor): Ground truth labels.
            y_pred (tensor): Predicted labels.

        Returns:
            tensor: IoU value.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
        return tf.reduce_mean((intersection + 1e-15) / (union + 1e-15), axis=0)
    
    def dice_loss(y_true, y_pred, smooth=1):
        """
        Computes the Dice loss function.

        Args:
            y_true (tensor): Ground truth labels.
            y_pred (tensor): Predicted labels.
            smooth (float): Smoothing factor to avoid division by zero.

        Returns:
            tensor: Dice loss value.
        """
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    # Load the model with custom objects
    model = tf.keras.models.load_model(model_path, custom_objects={'iou': iou, 'dice_loss': dice_loss})
    return model

def predict(model, segment):
    """
    Preprocesses the input image segment and makes a prediction using the loaded model.

    Args:
        model (tf.keras.Model): The loaded Keras model.
        segment (numpy.array): The input image segment to be processed.

    Returns:
        numpy.array: The model's prediction for the input image segment.
    """
    expected_size = (256, 256, 3)
    
    # Resize the image if it does not match the expected size
    if segment.shape != expected_size:
        segment = cv2.resize(segment, (expected_size[1], expected_size[0]))
    
    # Add a batch dimension
    segment = np.expand_dims(segment, axis=0)
    
    # Predict the segmentation mask
    prediction = model.predict(segment)
    return prediction[0]