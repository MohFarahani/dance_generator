import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from model_setup import Model_Setup

# https://analyticsindiamag.com/how-to-do-multivariate-time-series-forecasting-using-lstm/


class Train:
    def __init__(self, CONFIG):
        self.history = None
        self.input_size = None
        self.config = CONFIG

    def dataset_df(self, df):
        df_data = df.loc[:, ~df.columns.str.startswith("v")]
        df_data = df_data.iloc[:, 1:]

        return df_data

    def custom_ts_multi_data_prep(self, dataset, target, start, end):
        X = []
        y = []
        start = start + self.config.WINDOWS
        if end is None:
            end = len(dataset) - self.config.HORIZON
        for i in range(start, end):
            indices = range(i - self.config.WINDOWS, i)
            X.append(dataset.loc[indices, :])
            indicey = range(i + 1, i + 1 + self.config.HORIZON)
            y.append(target.loc[indicey, :])
        return np.array(X), np.array(y)

    def x_y_split(self, x_data, y_data):
        end_train = int((1 - self.config.TRAIN_SPLIT) * len(x_data))
        x_train, y_train = self.custom_ts_multi_data_prep(x_data, y_data, 0, end_train)
        x_vali, y_vali = self.custom_ts_multi_data_prep(x_data, y_data, end_train, None)

        return x_train, y_train, x_vali, y_vali

    def train_val_data(self, df):
        # split data
        df_data = self.dataset_df(df)
        x_train, y_train, x_vali, y_vali = self.x_y_split(df_data, df_data)
        self.input_size = x_train.shape[-2:]
        # prepare trainig dataseet
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = (
            train_data.cache()
            .shuffle(self.config.BUFFER_SIZE)
            .batch(self.config.BATCH_SIZE)
            .repeat()
        )
        # prepare validation dataseet
        val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
        val_data = val_data.batch(self.config.BATCH_SIZE).repeat()
        return train_data, val_data

    def model(self):
        # NN architecture
        lstm_model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(256, return_sequences=True),
                    input_shape=self.input_size,
                ),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(units=self.config.HORIZON),
            ]
        )
        lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
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
            epochs=self.config.EPOCHS,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            validation_data=val_data,
            validation_steps=self.config.VALIDATION_STEPS,
            verbose=self.config.VERBOSE,
            callbacks=callbacks,
        )

    def plot_performance(self):
        plt.plot(self.history.history["loss"], label="loss")
        plt.plot(self.history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(self.history.history["rmse"], label="rmse")
        plt.plot(self.history.history["val_rmse"], label="val_rmse")
        plt.legend()
        plt.show()
        plt.close()
