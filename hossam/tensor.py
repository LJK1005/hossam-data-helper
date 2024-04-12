import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def tf_create(
    dense: list = [],
    optimizer: str = "adam",
    loss: str = None,
    metrics: list = None,
) -> Sequential:

    if not dense or not loss or not metrics:
        raise ValueError("dense, loss, and metrics are required arguments")

    model = Sequential()

    for i, v in enumerate(iterable=dense):
        if "input_shape" in v:
            model.add(
                Dense(
                    units=v["units"],
                    input_shape=v["input_shape"],
                    activation=v["activation"],
                )
            )
        else:
            model.add(Dense(units=v["units"], activation=v["activation"]))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def tf_train(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 500,
    batch_size: int = 32,
    validation_data: tuple = None,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    verbose: int = 0,
) -> Sequential:

    callbacks = []

    if early_stopping:
        callbacks.append(
            EarlyStopping(patience=10, restore_best_weights=True, verbose=verbose)
        )

    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(factor=0.1, patience=5, verbose=verbose))

    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=verbose,
        callbacks=callbacks,
    )
