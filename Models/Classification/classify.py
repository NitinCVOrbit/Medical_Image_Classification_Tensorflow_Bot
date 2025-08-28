from PIL import ImageOps
import numpy as np
import tensorflow as tf
import os


def preprocessing(img):
    """Preprocesses an image for classification."""
    img = ImageOps.grayscale(img)
    img = img.resize((224, 224))
    img_pred = np.array(img)
    img_pred = np.expand_dims(img_pred, axis=0)
    img_pred = img_pred / 255.0
    return img_pred


def classifier(image, weights, class_names):    
    """Classifies an image as Brain Tumor or No Brain Tumor."""
    preprocessed_img = preprocessing(image)
    
    weights_path = os.path.join(os.path.dirname(__file__), "..", "Weights",  f"{weights}")
    model = tf.keras.models.load_model(weights_path)    
    
    pred = model.predict(preprocessed_img)[0]
    pred_index = np.argmax(pred, axis=-1)
    score = int(np.round(pred[pred_index], 4) * 100) 
    label = class_names[pred_index]
    
    return label,score