import tensorflow as tf
from tensorflow.keras import layers

def build_model():
    inputs = tf.keras.Input(shape=(128, 128, 3))

    # Normalize RGB values
    x = layers.Rescaling(1. / 255)(inputs)

    # Block 1
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="linear")(x)

    return tf.keras.Model(inputs, outputs)


# Custom metric to enforce top of the predictions
def top1_soft_2(y_true, y_pred, tol=0.05):  # tighten tol from 0.1 to 0.05
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    pred_idx = tf.argmax(y_pred)
    pred_true = tf.gather(y_true, pred_idx)
    max_true = tf.reduce_max(y_true)
    return tf.cast(pred_true >= (max_true - tol), tf.float32)


# During the making of prototype 11j I rested this loss function after removing the other element
# In doing so I didn't realize it was just a Huber loss function.
# So 11j just runs on Huber Loss but references this function
def combined_loss(y_true, y_pred):
    return tf.keras.losses.Huber(delta=0.1)(y_true, y_pred)