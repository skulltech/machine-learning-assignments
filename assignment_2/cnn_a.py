import argparse
import os
import sys
import time
import datetime

import numpy as np
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Activation
from keras.models import Sequential, load_model
from keras.optimizers import rmsprop, adam
from sklearn.preprocessing import LabelBinarizer


TTL = 3000
class LimitTrainingTime(Callback):
    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, batch, logs={}):
        time_elapsed = time.time() - self.start_time
        hr = str(datetime.timedelta(seconds=time_elapsed))
        print(f'[*] Time elapsed: {hr}')
        if time_elapsed > TTL:
            self.model.stop_training = True


def keras_cnn(args):
    with open(args.trainfile) as f:
        train = np.loadtxt(f, delimiter=' ')
    x = train[:, :-1]
    y = train[:, -1:]
    x = np.reshape(x, (x.shape[0], 32, 32, 3))
    lbl = LabelBinarizer()
    y = lbl.fit_transform(y)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', input_shape=x.shape[1:], kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(Dense(10, kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    opt = adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    lmt =  LimitTrainingTime()
    es = EarlyStopping(monitor='val_acc', patience=3)
    mc = ModelCheckpoint('checkpoint', monitor='val_acc', save_best_only=True)
    model.fit(x, y, validation_split=0.1, epochs=100, batch_size=128, callbacks=[lmt, es, mc])
    model = load_model('checkpoint')

    with open(args.testfile) as f:
        test = np.loadtxt(f, delimiter=' ')
    x = test[:, :-1]
    x = np.reshape(x, (x.shape[0], 32, 32, 3))
    
    probs = model.predict(x)
    preds = np.argmax(probs, axis=1)
    np.savetxt(args.outputfile, preds, fmt='%i')



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('trainfile', type=str)
    parser.add_argument('testfile', type=str)
    parser.add_argument('outputfile', type=str)
    parser.set_defaults(func=keras_cnn)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)



if __name__=='__main__':
    main()
