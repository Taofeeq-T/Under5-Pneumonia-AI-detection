import tensorflow as tf
from .build_model import build_vgg19_model

def train_model(train_dataset, validation_dataset, pooling, dropout, base_learning_rate, epochs=20):
    model = build_vgg19_model(pooling, dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stop]
    )
    return model, history
