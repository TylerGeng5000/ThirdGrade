import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ---------- global settings ---------- #
SEED = 42
NUM_CLASSES = 10
BASELINE_EPOCHS = 30
IMPROVED_EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
MODEL_DIR = "model"

tf.random.set_seed(SEED)
np.random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)


def one_hot(label, num_classes):
    labels = np.squeeze(label).astype("int32")
    label_one_hot = np.zeros((len(labels), num_classes), dtype="float32")
    label_one_hot[np.arange(len(labels)), labels] = 1.0
    return label_one_hot


def get_metric_keys(history_dict):
    train_acc_key = "accuracy" if "accuracy" in history_dict else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history_dict else "val_acc"
    return train_acc_key, val_acc_key


def plot_history(history, title_prefix):
    history_dict = history.history
    train_acc_key, val_acc_key = get_metric_keys(history_dict)

    plt.figure(figsize=(8, 5))
    plt.plot(np.array(history_dict["loss"]))
    plt.plot(np.array(history_dict["val_loss"]))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.legend(["loss", "val_loss"])
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(np.array(history_dict[train_acc_key]))
    plt.plot(np.array(history_dict[val_acc_key]))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} - Accuracy")
    plt.legend([train_acc_key, val_acc_key])
    plt.show()


def build_baseline_model():
    model = keras.Sequential(name="baseline_cnn")
    model.add(layers.Conv2D(32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    return model


def build_improved_model():
    model = keras.Sequential(name="improved_cnn")

    # 改进点1：激活函数改为swish并加入BN，提升深层训练稳定性。
    model.add(layers.Conv2D(64, (3, 3), padding="same", input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("swish"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("swish"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("swish"))
    model.add(layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("swish"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.30))

    model.add(layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("swish"))
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(256, activation="swish", kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    return model


def evaluate_and_show_predictions(model, x_test, y_test):
    test_pred_prob = model.predict(x_test, verbose=1)
    test_pred = np.argmax(test_pred_prob, axis=1)
    accuracy = np.mean(test_pred == y_test)
    print("test labels sample:", y_test[:3])
    print("predictions for first 3 samples:")
    print(model.predict(x_test[:3], verbose=0))
    print("accuracy =", float(accuracy))
    return float(accuracy)


def main():
    # ---------- 2.2 Load CIFAR-10 dataset ---------- #
    (x_train, y_train_raw), (x_test, y_test_raw) = cifar10.load_data()

    # ---------- 2.3 data preprocessing ---------- #
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # ---------- 2.4 label preprocessing ---------- #
    y_train = one_hot(y_train_raw, NUM_CLASSES)
    y_test = np.squeeze(y_test_raw).astype("int32")
    y_test_onehot = one_hot(y_test_raw, NUM_CLASSES)

    split_idx = int((1.0 - VALIDATION_SPLIT) * x_train.shape[0])
    x_train_main, x_val = x_train[:split_idx], x_train[split_idx:]
    y_train_main, y_val = y_train[:split_idx], y_train[split_idx:]

    # ---------- baseline experiment ---------- #
    baseline_model = build_baseline_model()
    print(baseline_model.summary())
    baseline_model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )

    history_baseline = baseline_model.fit(
        x_train_main,
        y_train_main,
        epochs=BASELINE_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1,
        validation_data=(x_val, y_val),
    )
    plot_history(history_baseline, "Baseline")
    baseline_model.save(os.path.join(MODEL_DIR, "cnn_baseline.h5"))
    baseline_test_loss, baseline_test_acc = baseline_model.evaluate(x_test, y_test_onehot, verbose=0)

    # ---------- improved experiment ---------- #
    improved_model = build_improved_model()
    print(improved_model.summary())

    # 改进点2：损失函数改为带label smoothing的交叉熵，提升泛化能力。
    improved_model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )

    augmenter = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="augmenter",
    )

    train_ds = tf.data.Dataset.from_tensor_slices((x_train_main, y_train_main))
    train_ds = train_ds.shuffle(buffer_size=10000, seed=SEED)
    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "cnn_improved.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history_improved = improved_model.fit(
        train_ds,
        epochs=IMPROVED_EPOCHS,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )
    plot_history(history_improved, "Improved")

    best_model = keras.models.load_model(os.path.join(MODEL_DIR, "cnn_improved.keras"))
    improved_test_loss, improved_test_acc = best_model.evaluate(x_test, y_test_onehot, verbose=0)
    _ = evaluate_and_show_predictions(best_model, x_test, y_test)

    print("\n===== Experiment Comparison =====")
    print(f"Baseline  : test_loss={baseline_test_loss:.4f}, test_acc={baseline_test_acc:.4f}")
    print(f"Improved  : test_loss={improved_test_loss:.4f}, test_acc={improved_test_acc:.4f}")
    print(f"Acc gain  : {improved_test_acc - baseline_test_acc:+.4f}")


if __name__ == "__main__":
    main()