"""
2-class CNN-LSTM for EEG motor imagery: Rest (0) vs Motor (1).
Input shape: (batch, channels=64, time=672). Output: 2 probabilities.
"""
import numpy as np


def build_model(n_channels=64, n_times=672, n_classes=2):
    """Build and return a Keras CNN-LSTM model (tuned for ≥90% accuracy)."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Install tensorflow: pip install tensorflow")

    reg = tf.keras.regularizers.L2(5e-5)
    inp = tf.keras.layers.Input(shape=(n_channels, n_times))
    # (batch, 64, 672) -> (batch, 672, 64) for Conv1D along time
    x = tf.keras.layers.Permute((2, 1))(inp)
    x = tf.keras.layers.Conv1D(32, 8, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 4, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.25)(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def to_two_class(labels_5):
    """Map 5-class labels to 2-class: 4 -> 0 (Rest), 0,1,2,3 -> 1 (Motor)."""
    return np.where(np.asarray(labels_5) == 4, 0, 1)


def to_three_class(labels_5):
    """Map 5-class to 3-class: 4->0 (Rest), 0->1 (Left Hand), 1->2 (Right Hand)."""
    labels = np.asarray(labels_5)
    out = np.zeros_like(labels, dtype=np.int64)
    out[labels == 4] = 0   # Rest
    out[labels == 0] = 1   # Left Hand
    out[labels == 1] = 2   # Right Hand
    return out


def balanced_accuracy_3class(y_true, y_pred):
    """Per-class recall averaged (3-class). Use for early stopping so motor classes matter."""
    import tensorflow as tf
    n_classes = 3
    y_pred_class = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
    y_pred_class = tf.reshape(y_pred_class, [-1])
    recalls = []
    for c in range(n_classes):
        in_c = tf.equal(y_true, c)
        n_c = tf.maximum(tf.reduce_sum(tf.cast(in_c, tf.float32)), 1.0)
        correct_c = tf.reduce_sum(tf.cast(tf.logical_and(in_c, tf.equal(y_pred_class, c)), tf.float32))
        recalls.append(correct_c / n_c)
    return tf.reduce_mean(recalls)


def build_model_3class(n_channels=64, n_times=672):
    """Build 3-class CNN-LSTM (Rest, Left Hand, Right Hand). Regularization tuned to avoid Rest-only collapse."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Install tensorflow: pip install tensorflow")

    reg = tf.keras.regularizers.L2(8e-5)
    inp = tf.keras.layers.Input(shape=(n_channels, n_times))
    # (batch, 64, 672) -> (batch, 672, 64)
    x = tf.keras.layers.Permute((2, 1))(inp)
    # Block 1
    x = tf.keras.layers.Conv1D(64, 12, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # Block 2
    x = tf.keras.layers.Conv1D(128, 6, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # Block 3
    x = tf.keras.layers.Conv1D(128, 4, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    # LSTM
    x = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", balanced_accuracy_3class],
    )
    return model
