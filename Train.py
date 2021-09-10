import matplotlib as plt
import numpy as np
import pandas as pd
import tensorflow as tf

HIST_WINDOW = 30
HORIZON = 1
TRAIN_SPLIT = 0.2

BATCH_SIZE = 256
BUFFER_SIZE = 150

EPOCHS = 150
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 50
VERBOSE = 1

# https://analyticsindiamag.com/how-to-do-multivariate-time-series-forecasting-using-lstm/


class Train:
    def __init__(self):
        self.history = None
        self.input_size = None

    def dataset_df(self,df):
        df_data = df.loc[:, ~df.columns.str.startswith("v")]
        df_data = df_data.iloc[:, 1:]

        return df_data

    def custom_ts_multi_data_prep(
        self, dataset, target, start, end, window=HIST_WINDOW, horizon=HORIZON
    ):
        X = []
        y = []
        start = start + window
        if end is None:
            end = len(dataset) - horizon
        for i in range(start, end):
            indices = range(i - window, i)
            X.append(dataset.loc[indices,:])
            indicey = range(i + 1, i + 1 + horizon)
            y.append(target.loc[indicey,:])
        return np.array(X), np.array(y)

    def x_y_split(self, x_data, y_data):
        end_train = int((1 - TRAIN_SPLIT) * len(x_data))
        x_train, y_train = self.custom_ts_multi_data_prep(
            x_data, y_data, 0, end_train, window=HIST_WINDOW, horizon=HORIZON
        )
        x_vali, y_vali = self.custom_ts_multi_data_prep(
            x_data, y_data, end_train, None, window=HIST_WINDOW, horizon=HORIZON
        )

        return x_train, y_train, x_vali, y_vali

    def train_val_data(self, df):
        # split data
        df_data = self.dataset_df(df)
        x_train, y_train, x_vali, y_vali = self.x_y_split(df_data, df_data)
        self.input_size = x_train.shape[-2:]
        # prepare trainig dataseet
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        # prepare validation dataseet
        val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
        val_data = val_data.batch(BATCH_SIZE).repeat()
        return train_data, val_data

    def model(self):
        # NN architecture
        lstm_model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(200, return_sequences=True),
                    input_shape=self.input_size,
                ),
                tf.keras.layers.Dense(100, activation="relu"),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(units=HORIZON),
            ]
        )
        lstm_model.compile(
            optimizer="adam", 
            loss="mse",
            metrics = ["accuracy"])
        lstm_model.summary()
        return lstm_model

    def fit(self, df):
        model_path = "Bidirectional_LSTM_Multivariate.h5"
        early_stopings = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="min"
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0
        )
        callbacks = [early_stopings, checkpoint]
        train_data, val_data = self.train_val_data(df)
        lstm_model = self.model()
        self.history = lstm_model.fit(
            train_data,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=val_data,
            validation_steps=VALIDATION_STEPS,
            verbose=VERBOSE,
            callbacks=callbacks,
        )

    def plot_performance(self):
        plt.figure(figsize=(16, 9))
        plt.plot(self.history.history["loss"], label = "loss")
        plt.plot(self.history.history["val_loss"], label = "val_loss")
        plt.legend()
        plt.show()
        plt.close()
        plt.figure(figsize=(16, 9))
        plt.plot(self.history.history["accuracy"], label = "accuracy")
        plt.plot(self.history.history["val_accuracy"], label = "val_accuracy")
        plt.legend()
        plt.show()
        plt.close()