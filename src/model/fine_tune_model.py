import tensorflow as tf

def fine_tune_model(train_dataset, validation_dataset, model_path, initial_lr, epochs=50):
    model = tf.keras.models.load_model(model_path)
    base_model = model.layers[1]
    base_model.trainable = True

    for layer in base_model.layers[:15]:  # Freeze first 15 layers
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr / 100),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stop]
    )
    return model, history
