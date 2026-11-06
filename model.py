
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def create_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.20,
            kernel_regularizer=regularizers.l2(1e-4),
            recurrent_regularizer=regularizers.l2(1e-4)
        )
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=False,
            dropout=0.20,
            kernel_regularizer=regularizers.l2(1e-4),
            recurrent_regularizer=regularizers.l2(1e-4)
        )
    )(x)

    x = layers.Dropout(0.30)(x)

    return models.Model(
        inputs=inputs,
        outputs=x,
        name="Regularized_BiLSTM_Encoder"
    )


def create_model(input_shape=(16, 119), num_classes=11):
    encoder = create_encoder(input_shape)

    model_input = layers.Input(
        shape=input_shape,
        name="FFT_Input"
    )

    encoded = encoder(model_input)

    output = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(1e-4),
        name="Activity_Classifier"
    )(encoded)

    model = models.Model(
        inputs=model_input,
        outputs=output,
        name="Regularized_BiLSTM"
    )

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, X_train, y_train):
    early_stop = EarlyStopping(
        monitor="loss",
        patience=10,
        min_delta=0.001,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        shuffle=True,
        callbacks=[early_stop],
        verbose=1
    )

    return history
