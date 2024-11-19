import tensorflow as tf

BATCH_SIZE = 64
IMG_SIZE = (224, 224)

def load_dataset(directory, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode="binary", seed=334):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        label_mode=label_mode,
        seed=seed
    )
