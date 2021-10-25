import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import copy
import h5py
from tensorflow.python.framework import config
import tensorflow_io as tfio
from tensorflow.keras import layers
from tensorflow import keras
import keras.backend as K
import numpy as np
from model_setup import Model_Setup
# https://analyticsindiamag.com/how-to-do-multivariate-time-series-forecasting-using-lstm/
class Train:
    def __init__(self, CONFIG):
        self.history = None
        self.input_size = None
        self.config = CONFIG
        self.past_size = CONFIG.HIST_WINDOW 
        self.features = 3*CONFIG.NUM_COORDS

    def dataset_df(self, df):
        df_data = df.loc[:, ~df.columns.str.startswith("v")]
        df_data = df_data.iloc[:, 1:]
        if self.config.SIMPLE_MODEL:
            drop_id = [1,2,3,4,6,7,8,9,10,19,20,21,22]
            for id in drop_id:
                df_data = df_data.loc[:, ~df_data.columns.str.endswith("x{}".format(id))]
                df_data = df_data.loc[:, ~df_data.columns.str.endswith("y{}".format(id))]
                df_data = df_data.loc[:, ~df_data.columns.str.endswith("z{}".format(id))]

        self.features = len(df_data.columns)
        return df_data

    def custom_ts_multi_data_prep(self, dataset, target, start, end):
        X = []
        y = []
        start = start + self.config.HIST_WINDOW
        if end is None:
            end = len(dataset) - self.config.HORIZON
        for i in range(start, end):
            indices = range(i - self.config.HIST_WINDOW, i)
            X.append(dataset.loc[indices, :])
            indicey = range(i + 1, i + 1 + self.config.HORIZON)
            y.append(target.loc[indicey, :])
        return X, y

    def custom_ts_multi_data_prep_multi(self, X, y, dataset, target, start, end):
        start = start + self.config.HIST_WINDOW
        if end is None:
            end = len(dataset) - self.config.HORIZON
        for i in range(start, end):
            indices = range(i - self.config.HIST_WINDOW, i)
            X.append(dataset.loc[indices, :])
            indicey = range(i + 1, i + 1 + self.config.HORIZON)
            y.append(target.loc[indicey, :])
        return X, y

    def x_y_split(self, x_data, y_data):
        end_train = int((1 - self.config.TRAIN_SPLIT) * len(x_data))
        x_train, y_train = self.custom_ts_multi_data_prep(x_data, y_data, 0, end_train)
        x_vali, y_vali = self.custom_ts_multi_data_prep(x_data, y_data, end_train, None)

        return x_train, y_train, x_vali, y_vali

    def x_y_split_multifile(self, RESULTS_CSV):
        for csvfile in os.listdir(RESULTS_CSV):
            if csvfile.endswith(".csv"):
                x_train, y_train, x_vali, y_vali = [], [], [], []
                csvfile = RESULTS_CSV + "/" + csvfile
                df_data = pd.read_csv(csvfile)
                df_data = self.dataset_df(df_data)
                #df_data = df_data[df_data.index % self.config.KEEP_FRAME==0]
                end_train = int((1 - self.config.TRAIN_SPLIT) * len(df_data))

                x_train, y_train = self.custom_ts_multi_data_prep_multi(
                    x_train, y_train, df_data, df_data, 0, end_train
                )
                x_vali, y_vali = self.custom_ts_multi_data_prep_multi(
                    x_vali, y_vali, df_data, df_data, end_train, None
                )
                x_train = np.array(x_train)
                y_train = np.array(y_train)
                x_vali = np.array(x_vali)
                y_vali = np.array(y_vali)
                print(csvfile)
                if  os.path.isfile(self.config.HDF) == False:
                    with h5py.File(self.config.HDF, 'w') as hf:
                        hf.create_dataset('x_train', data=x_train,compression="gzip", chunks=True, maxshape=(None,self.config.HIST_WINDOW,3*self.config.NUM_COORDS))
                        hf.create_dataset('y_train', data=y_train,compression="gzip", chunks=True, maxshape=(None,self.config.HORIZON,3*self.config.NUM_COORDS))
                        hf.create_dataset('x_vali', data=x_vali,compression="gzip", chunks=True, maxshape=(None,self.config.HIST_WINDOW,3*self.config.NUM_COORDS))
                        hf.create_dataset('y_vali', data=y_vali,compression="gzip", chunks=True, maxshape=(None,self.config.HORIZON,3*self.config.NUM_COORDS))

                else:
                    with h5py.File(self.config.HDF, 'a') as hf:
                        hf["x_train"].resize((hf["x_train"].shape[0] + x_train.shape[0]), axis = 0)
                        hf["x_train"][-x_train.shape[0]:] = x_train

                        hf["y_train"].resize((hf["y_train"].shape[0] + y_train.shape[0]), axis = 0)
                        hf["y_train"][-y_train.shape[0]:] = y_train

                        hf["x_vali"].resize((hf["x_vali"].shape[0] + x_vali.shape[0]), axis = 0)
                        hf["x_vali"][-x_vali.shape[0]:] = x_vali

                        hf["y_vali"].resize((hf["y_vali"].shape[0] + y_vali.shape[0]), axis = 0)
                        hf["y_vali"][-y_vali.shape[0]:] = y_vali

        return np.array(x_train), np.array(y_train), np.array(x_vali), np.array(y_vali)

    def train_val_data(self, RESULT_CSV):

        # split data
        x_train, y_train, x_vali, y_vali = self.x_y_split_multifile(RESULT_CSV)

        # prepare trainig dataseet
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = (
            train_data.cache()
            .shuffle(self.config.BUFFER_SIZE)
            .batch(self.config.BATCH_SIZE)
        )
        # prepare validation dataseet
        val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
        val_data = val_data.batch(self.config.BATCH_SIZE)
        return train_data, val_data

    def train_val_data_HDF(self, RESULT_CSV):
        self.x_y_split_multifile(RESULT_CSV)
        # Create an IODataset from a hdf5 file's dataset object  
        x_train = tfio.IODataset.from_hdf5(self.config.HDF, dataset='/x_train')
        y_train = tfio.IODataset.from_hdf5(self.config.HDF, dataset='/y_train')
        x_vali = tfio.IODataset.from_hdf5(self.config.HDF, dataset='/x_vali')
        y_vali= tfio.IODataset.from_hdf5(self.config.HDF, dataset='/y_vali')
        # Zip together samples and corresponding labels
        train_data = tf.data.Dataset.zip((x_train,y_train)).batch(self.config.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).repeat()
        val_data = tf.data.Dataset.zip((x_vali,y_vali)).batch(self.config.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).repeat()
        return train_data,val_data

    def model(self):
        # NN architecture
        if self.config.MODEL_NAME == "autoregression":
            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(256, return_sequences=True),
                        input_shape=(self.past_size, self.features),
                    ),
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(128, return_sequences=False),
                        input_shape=(self.past_size, self.features),
                    ),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(256, activation="relu"),
                    tf.keras.layers.Dense(self.features),
                ]
            )
        elif self.config.MODEL_NAME == "autoencoder":
            # https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-2-0-keras/
            encoder_inputs = tf.keras.layers.Input(
                shape=(self.past_size, self.features)
            )
            encoder_l1 = tf.keras.layers.LSTM(
                256, return_sequences=True, return_state=True
            )
            encoder_outputs1 = encoder_l1(encoder_inputs)
            encoder_states1 = encoder_outputs1[1:]
            encoder_l2 = tf.keras.layers.LSTM(256, return_state=True)
            encoder_outputs2 = encoder_l2(encoder_outputs1[0])
            encoder_states2 = encoder_outputs2[1:]
            #
            decoder_inputs = tf.keras.layers.RepeatVector(self.config.HORIZON)(
                encoder_outputs2[0]
            )
            #
            decoder_l1 = tf.keras.layers.LSTM(256, return_sequences=True)(
                decoder_inputs, initial_state=encoder_states1
            )
            decoder_l2 = tf.keras.layers.LSTM(256, return_sequences=False)(
                decoder_l1, initial_state=encoder_states2
            )
            decoder_outputs2 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.features)
            )(decoder_l2)
            #
            model = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
        elif self.config.MODEL_NAME == "lstm":
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.LSTM(256, return_sequences=True,
                    input_shape=(self.past_size, self.features)
                    ))
            model.add(tf.keras.layers.LSTM(256, return_sequences=False
                    ))
            model.add(tf.keras.layers.Dense(1024, activation="relu"))
            model.add(tf.keras.layers.Dropout(0.2))                 
            model.add(tf.keras.layers.Dense(512, activation="relu"))
            model.add(tf.keras.layers.Dropout(0.2))              
            model.add(tf.keras.layers.Dense(256, activation="relu"))
            model.add(tf.keras.layers.Dropout(0.1))                  
            #model.add(tf.keras.layers.Dense(128, activation="relu"))
            if self.config.PROBABILISTIC:
                model.add(tf.keras.layers.Dense(2*self.features))
            else:
                model.add(tf.keras.layers.Dense(self.features))
            
        elif self.config.MODEL_NAME=="multihead_attention":
            # https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb
            # https://keras.io/examples/vision/image_classification_with_vision_transformer/
            def mlp(x, hidden_units, dropout_rate):
                for units in hidden_units:
                    x = layers.Dense(units, activation=tf.nn.gelu)(x)
                    x = layers.Dropout(dropout_rate)(x)
                return x
            inputs = layers.Input(shape=(self.past_size,self.features))

            input_multihead = tf.identity(inputs)
            # Create multiple layers of the Transformer block.
            for _ in range(self.config.TRANSFORMER_LAYERS):
                # Layer normalization 1.
                x1 = layers.LayerNormalization(epsilon=1e-6)(input_multihead)
                # Create a multi-head attention layer.
                attention_output = layers.MultiHeadAttention(
                    num_heads=self.config.NUM_HEADS, key_dim=self.config.PROJECTION_DIM, dropout=0.1
                )(x1, x1)
                # Skip connection 1.
                x2 = layers.Add()([attention_output, input_multihead])
                # Layer normalization 2.
                x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
                # MLP.
                x3 = mlp(x3, hidden_units=self.config.TRANSFORMER_UNITS, dropout_rate=0.1)
                # Skip connection 2.
                input_multihead = layers.Add()([x3, x2])

            # Create a [batch_size, projection_dim] tensor.
            representation = layers.LayerNormalization(epsilon=1e-6)(input_multihead)
            representation = layers.Flatten()(representation)
            representation = layers.Dropout(0.5)(representation)
            # Add MLP.
            features = mlp(representation, hidden_units=self.config.MLP_HEAD_UNITS, dropout_rate=0.5)
            # outputs
            outputs = layers.Dense(self.features)(features)
            # Create the Keras model.
            model = keras.Model(inputs=inputs, outputs=outputs)
        elif self.config.MODEL_NAME == 'self_attention':
            # https://www.kaggle.com/danofer/covid-rna-lstm-self-attention-keras-model
            inputs = layers.Input(shape=(self.past_size,self.features))
            x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
            x = Attention(return_sequences=True)(x)
            x = keras.layers.LSTM(256,return_sequences=False)(x)
            x = keras.layers.Dense(256, activation="relu")(x)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.Dense(128, activation="relu")(x)
            outputs = keras.layers.Dense(self.features)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            #model = keras.models.Sequential()
            #model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True,input_shape=(self.past_size, self.features))))
            #model.add(Attention(return_sequences=True)) 
            #model.add(keras.layers.LSTM(256,return_sequences=False))
            #model.add(tf.keras.layers.Dense(256, activation="relu"))
            #model.add(tf.keras.layers.Dropout(0.2))                   
            #model.add(tf.keras.layers.Dense(128, activation="relu"))
            #model.add(tf.keras.layers.Dense(self.features))
            #model.build(input_shape=(None,self.past_size, self.features))         
            
        elif self.config.MODEL_NAME == "custom":
            model = self.config.MODEL

        if self.config.PROBABILISTIC: 
            model.compile(optimizer="adam", loss=gaussian_nll)
        else:   
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.summary()

        return model

    def model_multi_parts(self):
        # https://google.github.io/mediapipe/solutions/pose.html
        head = 11*3        #0-11
        right_hand  = 6*3  #12-22
        left_hand   = 6*3  #11-21
        right_leg   = 5*3  #24-32
        left_leg    = 5*3  #23-31
        #
        inputs = layers.Input(shape=(self.past_size,self.features))
        # Head model
        #x_head = layers.LSTM(128, return_sequences=True,input_shape=(self.past_size,self.features))(inputs)
        #inputs_head = layers.Input(shape=(self.past_size,head))
        #x_head = layers.LSTM(50, return_sequences=True,input_shape=(self.past_size,head))(inputs_head)
        #x_head = layers.LSTM(128, return_sequences=False)(x_head)
        #x_head = layers.Dense(128, activation="relu")(x_head)
        #x_head_out = layers.Dense(head)(x_head)
        x_head = layers.Bidirectional(layers.LSTM(256, return_sequences=True,input_shape=(self.past_size,self.features)))(inputs)
        x_head = layers.LSTM(256, return_sequences=False)(x_head)
        x_head = layers.Dense(512, activation="relu")(x_head)
        x_head = layers.Dropout(0.4)(x_head)
        x_head = layers.Dense(128, activation="relu")(x_head)
        x_head_out = layers.Dense(head)(x_head)
        # Right Hand Model
        #x_right_hand = layers.LSTM(128, return_sequences=True,input_shape=(self.past_size,self.features))(inputs)
        #inputs_right_hand = layers.Input(shape=(self.past_size,right_hand))
        #x_right_hand = layers.LSTM(50, return_sequences=True,input_shape=(self.past_size,head))(inputs_right_hand)
        #x_right_hand = layers.LSTM(100, return_sequences=False)(x_right_hand)
        #x_right_hand = layers.Dense(128, activation="relu")(x_right_hand)
        x_right_hand = layers.Bidirectional(layers.LSTM(256, return_sequences=True,input_shape=(self.past_size,self.features)))(inputs)
        x_right_hand = layers.LSTM(256, return_sequences=False)(x_right_hand)
        x_right_hand = layers.Dense(512, activation="relu")(x_right_hand)
        x_right_hand = layers.Dropout(0.4)(x_right_hand)
        x_right_hand = layers.Dense(128, activation="relu")(x_right_hand)
        x_right_hand_out = layers.Dense(right_hand)(x_right_hand)
        # Left Hand Model
        #x_left_hand = layers.LSTM(128, return_sequences=True,input_shape=(self.past_size,self.features))(inputs)
        #inputs_left_hand = layers.Input(shape=(self.past_size,left_hand))
        #x_left_hand = layers.LSTM(50, return_sequences=True,input_shape=(self.past_size,head))(inputs_left_hand)
        #x_left_hand = layers.LSTM(100, return_sequences=False)(x_left_hand)
        #x_left_hand = layers.Dense(128, activation="relu")(x_left_hand)
        x_left_hand = layers.Bidirectional(layers.LSTM(256, return_sequences=True,input_shape=(self.past_size,self.features)))(inputs)
        x_left_hand = layers.LSTM(256, return_sequences=False)(x_left_hand)
        x_left_hand = layers.Dense(512, activation="relu")(x_left_hand)
        x_left_hand = layers.Dropout(0.4)(x_left_hand)
        x_left_hand = layers.Dense(128, activation="relu")(x_left_hand)
        x_left_hand_out = layers.Dense(left_hand)(x_left_hand)
        # Right Leg Model
        #x_right_leg = layers.LSTM(128, return_sequences=True,input_shape=(self.past_size,self.features))(inputs)
        #inputs_right_leg = layers.Input(shape=(self.past_size,right_leg))
        #x_right_leg = layers.LSTM(50, return_sequences=True,input_shape=(self.past_size,head))(inputs_right_leg)
        #x_right_leg = layers.LSTM(100, return_sequences=False)(x_right_leg)
        #x_right_leg = layers.Dense(128, activation="relu")(x_right_leg)
        x_right_leg = layers.Bidirectional(layers.LSTM(256, return_sequences=True,input_shape=(self.past_size,self.features)))(inputs)
        x_right_leg = layers.LSTM(256, return_sequences=False)(x_right_leg)
        x_right_leg = layers.Dense(512, activation="relu")(x_right_leg)
        x_right_leg = layers.Dropout(0.4)(x_right_leg)
        x_right_leg = layers.Dense(128, activation="relu")(x_right_leg)
        x_right_leg_out = layers.Dense(right_leg)(x_right_leg)
        # Left Leg Model
        #x_left_leg = layers.LSTM(128, return_sequences=True,input_shape=(self.past_size,self.features))(inputs)        
        #inputs_left_leg = layers.Input(shape=(self.past_size,left_leg))
        #x_left_leg = layers.LSTM(50, return_sequences=True,input_shape=(self.past_size,head))(inputs_left_leg)
        #x_left_leg = layers.LSTM(100, return_sequences=False)(x_left_leg)
        #x_left_leg = layers.Dense(128, activation="relu")(x_left_leg)
        x_left_leg = layers.Bidirectional(layers.LSTM(256, return_sequences=True,input_shape=(self.past_size,self.features)))(inputs)
        x_left_leg = layers.LSTM(256, return_sequences=False)(x_left_leg)
        x_left_leg = layers.Dense(512, activation="relu")(x_left_leg)
        x_left_leg = layers.Dropout(0.4)(x_left_leg)
        x_left_leg = layers.Dense(128, activation="relu")(x_left_leg)
        x_left_leg_out = layers.Dense(left_leg)(x_left_leg)

        #model = keras.Model(input=[inputs_head,inputs_right_hand,inputs_left_hand,inputs_right_leg,inputs_left_leg],outputs=[x_head_out,x_right_hand_out,x_left_hand_out,x_right_leg_out,x_left_leg_out])
        model = keras.Model(inputs=inputs,outputs=[x_head_out,x_right_hand_out,x_left_hand_out,x_right_leg_out,x_left_leg_out])

        model.compile(optimizer="adam", loss=["mse","mse","mse","mse","mse"], metrics=[["mae"],["mae"],["mae"],["mae"],["mae"]])
        model.summary()

        return model

    def fit(self, RESULT_CSV):
        model_path = self.config.MODEL_NAME +'.hdf5'
        early_stopings = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="min"
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0
        )
        callbacks = [early_stopings, checkpoint]
        #train_data, val_data = self.train_val_data_HDF(RESULT_CSV)
        if self.config.CREATE_HDF:
            self.x_y_split_multifile(RESULT_CSV)
        
        model = self.model()

        hdf = h5py.File(self.config.HDF, "r")
        
        train_generator = Data_Generator(hdf['x_train'],hdf['y_train'],self.config.BATCH_SIZE)
        vali_generator  = Data_Generator(hdf['x_vali'],hdf['y_vali'],self.config.BATCH_SIZE)

        self.history = model.fit(
            train_generator,
            batch_size = self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            validation_data=vali_generator,
            #validation_steps=self.config.VALIDATION_STEPS,
            verbose=self.config.VERBOSE,
            callbacks=callbacks,
            )
        hdf.close()

    def fit_multi_parts(self, RESULT_CSV):
        model_path = self.config.MODEL_NAME +'.hdf5'
        early_stopings = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="min"
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0
        )
        callbacks = [early_stopings, checkpoint]
        #train_data, val_data = self.train_val_data_HDF(RESULT_CSV)
        if self.config.CREATE_HDF:
            self.x_y_split_multifile(RESULT_CSV)
        model = self.model_multi_parts()

        hdf = h5py.File(self.config.HDF, "r")
        
        train_generator = Data_Generator_multi_parts(hdf['x_train'],hdf['y_train'],self.config.BATCH_SIZE)
        vali_generator  = Data_Generator_multi_parts(hdf['x_vali'],hdf['y_vali'],self.config.BATCH_SIZE)

        self.history = model.fit(
            train_generator,
            batch_size = self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            validation_data=vali_generator,
            #validation_steps=self.config.VALIDATION_STEPS,
            verbose=self.config.VERBOSE,
            callbacks=callbacks,
            )
        hdf.close()

    def plot_performance(self):
        plt.plot(self.history.history["loss"], label="loss")
        plt.plot(self.history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(self.history.history["mae"], label="mae")
        plt.plot(self.history.history["val_mae"], label="val_mae")
        plt.legend()
        plt.show()
        plt.close()

    def generator(self, MODEL_PATH, df_init, frames_future):
        model = tf.keras.models.load_model(MODEL_PATH+'.hdf5', compile=False)
        model.summary()
        for _ in range(frames_future):
            x = df_init.tail(self.config.HIST_WINDOW)
            x = np.array(x)
            x = np.expand_dims(x, axis=0)
            prediction = model.predict(x)
            if self.config.PROBABILISTIC: 
                n_outs = int(prediction.shape[1] / 2)
                mean = prediction[:, 0:n_outs]
                sigma = np.exp(prediction[:, n_outs:])
                prediction = np.random.normal(mean, sigma)
            if len(prediction.shape) ==3:
                prediction = prediction[0,:,:]
            data_to_append = pd.DataFrame(prediction, columns=df_init.columns)
            df_init = df_init.append(data_to_append, ignore_index=True)
        df_init.to_csv("generate.csv")

    def generator_multi_parts(self, MODEL_PATH, df_init, frames_future):
        model = tf.keras.models.load_model(MODEL_PATH+'.hdf5')
        model.summary()
        for _ in range(frames_future):
            x = df_init.tail(self.config.HIST_WINDOW)
            x = np.array(x)
            x = np.expand_dims(x, axis=0)
            head,right_hand,left_hand,right_leg,left_leg = model.predict(x)
            prediction = np.array([self.features*[0.0]])
            prediction[:,head_id] = head[0,:]
            prediction[:,right_hand_id] = right_hand[0,:]
            prediction[:,left_hand_id] = left_hand[0,:]
            prediction[:,right_leg_id] = right_leg[0,:]
            prediction[:,left_leg_id] = left_leg[0,:]
            data_to_append = pd.DataFrame(prediction, columns=df_init.columns)
            df_init = df_init.append(data_to_append, ignore_index=True)
        df_init.to_csv("generate.csv")


class Data_Generator(keras.utils.Sequence) :
  
  def __init__(self, x_hdf, y_hdf, batch_size,shuffle=True) :
    
    self.x = x_hdf
    self.y = y_hdf
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end()    
    
  def __len__(self) :
    return (np.ceil(len(self.x) / float(self.batch_size))).astype(np.int)
  
  def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
  
  def __getitem__(self, idx) :
    batch_x = self.x[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return (batch_x,np.reshape(batch_y, (batch_y.shape[0]*batch_y.shape[1], batch_y.shape[2])))

class Data_Generator_multi_parts(keras.utils.Sequence) :
  
  def __init__(self, x_hdf, y_hdf, batch_size,shuffle=True) :
    
    self.x = x_hdf
    self.y = y_hdf
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end() 
    
  def __len__(self) :
    return (np.ceil(len(self.x) / float(self.batch_size))).astype(np.int)
  
  def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

  def __getitem__(self, idx) :
    batch_x = self.x[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = np.reshape(batch_y, (batch_y.shape[0]*batch_y.shape[1], batch_y.shape[2]))
    batch_y_head = batch_y[:,head_id]
    batch_y_right_hand = batch_y[:,right_hand_id]   
    batch_y_left_hand = batch_y[:,left_hand_id]   
    batch_y_right_leg = batch_y[:,right_leg_id]   
    batch_y_left_leg = batch_y[:,left_leg_id]  
    return batch_x,[batch_y_head,batch_y_right_hand,batch_y_left_hand,batch_y_right_leg,batch_y_left_leg]

class Attention(layers.Layer):
    # https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
    def __init__(self, return_sequences=True, name=None, **kwargs):
        super(Attention, self).__init__(name=name)
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                           initializer="glorot_uniform", trainable=True)
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                           initializer="glorot_uniform", trainable=True)
    
        super(Attention, self).build(input_shape)
        
    def call(self, x):
        
        e = keras.backend.tanh(keras.backend.dot(x,self.W)+self.b)
        a = keras.backend.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return keras.backend.sum(output, axis=1)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
#             'vocab_size': self.vocab_size,
            'num_layers': 2, # self.num_layers,
            'units': 200, #self.units,
            #'d_model': self.d_model,
            'num_heads': 2 ,# self.num_heads,
            'dropout': 0.25 , #self.dropout,
        })
        return config

def calc_id():
    head = [0,1,2,3,4,5,6,7,8,9,10]        #0-11
    head_id = 3*len(head)*[0]
    i = 0
    for value in range(0,11):
        head_id[i] = 3*value
        head_id[i+1] = 3*value + 1
        head_id[i+2] = 3*value + 2
        i += 3
          
    right_hand  = [12,14,16,18,20,22]  #12-22
    right_hand_id = 3*len(right_hand)*[0]
    i = 0
    for value in range(12,24,2):
        right_hand_id[i] = 3*value
        right_hand_id[i+1] = 3*value + 1
        right_hand_id[i+2] = 3*value + 2
        i += 3
    
    left_hand   = [11,13,15,17,19,21] #11-21
    left_hand_id = 3*len(left_hand)*[0]
    i = 0
    for value in range(11,23,2):
        left_hand_id[i] = 3*value
        left_hand_id[i+1] = 3*value + 1
        left_hand_id[i+2] = 3*value + 2
        i += 3

    right_leg   = [24,26,28,30,32]  #24-32
    right_leg_id = 3*len(right_leg)*[0]
    i = 0
    for value in range(24,34,2):
        right_leg_id[i] = 3*value
        right_leg_id[i+1] = 3*value + 1
        right_leg_id[i+2] = 3*value + 2
        i += 3

    left_leg    = [23,25,27,29,31]  #23-31 
    left_leg_id = 3*len(left_leg)*[0]
    i = 0
    for value in range(23,33,2):
        left_leg_id[i] = 3*value
        left_leg_id[i+1] = 3*value + 1
        left_leg_id[i+2] = 3*value + 2
        i += 3

    return head_id,right_hand_id,left_hand_id,right_leg_id,left_leg_id

head_id,right_hand_id,left_hand_id,right_leg_id,left_leg_id = calc_id()


def gaussian_nll(ytrue, ypreds):
    # https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e
    # https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """
    
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]
    
    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi
    
    return K.mean(-log_likelihood)
