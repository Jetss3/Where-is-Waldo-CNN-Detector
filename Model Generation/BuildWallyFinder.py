import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from ReadData import read_data
from tensorflow.keras import layers
from BatchGenerator import image_batch_generator
from Model import build_model, top1_soft_2, combined_loss
from SupportFunctions import plot_history



if __name__ == "__main__":
    # Acquire data from Formated Labeled data
    data = read_data('../s11_bboxes', '../s11_images')

    train_wally, val_wally = train_test_split(data, test_size=.2)
    sub_w = 128
    sub_h = 128
    batch_size = 64

    # Connect generator to dataset when calling batch
    train_ds = tf.data.Dataset.from_generator(
        lambda: image_batch_generator(train_wally, num_crops=batch_size, sub_w=sub_w, sub_h=sub_h),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, sub_w, sub_h, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
        )
    )

    # Ensure training data infinitely replenishes.
    # Redundant due to random data selection
    train_ds = train_ds.repeat()

    # Fixed validation dataset
    val_images = []
    val_labels = []

    # Create a generator for the validation dataset
    gen = image_batch_generator(val_wally, num_crops=batch_size, sub_w=sub_w, sub_h=sub_h)

    # Grab 20 batches
    for _ in range(20):
        img, labels = next(gen)
        val_images.append(img)
        val_labels.append(labels)

    # Join sequences
    val_images = np.concatenate(val_images)
    val_labels = np.concatenate(val_labels)

    # Save as tensorflow.data.Dataset
    val_ds_clean = tf.data.Dataset.from_tensor_slices(
        (val_images, val_labels)
    ).batch(batch_size)


    # Load model framework
    model = build_model()


    # Compile model with learning rate > 1e-4, else training will cost more than 300 epochs
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4, #
            weight_decay=1e-4),
        loss=combined_loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            top1_soft_2,
        ]
    )


    # Fit the model to ideal epochs, as per the 11j loss plot
    history = model.fit(
        train_ds,
        validation_data=val_ds_clean,
        epochs=130,
        steps_per_epoch=200,
    )


    # Save to file. !!! Ensure unique file before running !!!
    model.save("prototype11j.keras")


    # Evaluate on validation data
    results = model.evaluate(
        val_ds_clean,
        steps=50
    )


    # Plot history
    plot_history(history)

