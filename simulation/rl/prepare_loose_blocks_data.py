import csv
from cv.blocks_calibration import read_block_sizes
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scipy import stats
from collections import Counter
from random import random, choice, sample, choices


def calculate_block_height_params():
    # prepare dataset
    block_sizes = read_block_sizes('/home/bch_svt/cartpole/simulation/cv/block_sizes.json')

    heights = []
    for id in block_sizes:
        heights.append(block_sizes[id]['height'])

    mean = np.mean(heights)
    std = np.std(heights)

    return mean, std

def read_data():
    # prepare dataset
    block_sizes = read_block_sizes('/home/bch_svt/cartpole/simulation/cv/block_sizes.json')

    dataset = []

    with open(f"loose_blocks_data.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            row = list(map(int, row))
            row[0] = block_sizes[row[0]]['height']
            row[1] = block_sizes[row[1]]['height']
            row[2] = block_sizes[row[2]]['height']
            # print(f"Row: {row}")
            if row[0] > row[2]:
                row[0], row[2] = row[2], row[0]
                if row[3] == 1:
                    row[3] = 2
                if row[3] == 2:
                    row[3] = 1
            if row[3] == 3:
                row[3] = 2
            if row[3] == 4:
                row[3] = 3

            # print(row)
            dataset.append(row)

    dataset = np.reshape(dataset, (len(dataset), 4))

    x = dataset[:, :3]
    y = dataset[:, 3]

    return x, y

def prepare_data(x, y, mean=None, std=None):
    # shuffle
    y = np.reshape(y, (len(y), 1))
    dataset = np.concatenate((x, y), axis=1)
    np.random.shuffle(dataset)

    x = dataset[:, :3]
    y = dataset[:, 3]

    # noramlize
    if mean is None or std is None:
        mean = np.mean(x)
        std = np.std(x)

    x = (x - mean) / std

    # convert to one-hot encoding
    y = to_categorical(y, 4)

    return x, y, mean, std

def augment_data(x, y):
    category_0 = []
    category_1 = []
    category_2 = []
    category_3 = []

    augmented_category_0 = []
    augmented_category_1 = []
    augmented_category_2 = []
    augmented_category_3 = []

    for i in range(len(y)):
        if y[i] == 0:
            category_0.append(x[i])
        if y[i] == 1:
            category_1.append(x[i])
        if y[i] == 2:
            category_2.append(x[i])
        if y[i] == 3:
            category_3.append(x[i])

    max_elem_per_category = 1000

    # augment category 0
    while len(augmented_category_0) < max_elem_per_category - len(category_0):
        sample = choice(category_0)

        # add an equal amount to each height
        d_h = random() * 0.4 - 0.2
        sample = sample + d_h
        augmented_category_0.append(sample)

    # augment category 1
    while len(augmented_category_1) < max_elem_per_category - len(category_1):
        sample = choice(category_1)

        # subtract height from loose block and add equal amount to others
        d_h_positive = 0.2 * random()
        d_h_negative = -0.2 * random()
        sample[0] += d_h_negative
        sample[1:] += d_h_positive

        augmented_category_1.append(sample)

    # augment category 2
    while len(augmented_category_2) < max_elem_per_category - len(category_2):
        sample = choice(category_2)

        # subtract from loose blocks and add to other
        d_h_positive = 0.2 * random()
        d_h_negative1 = -0.2 * random()
        d_h_negative2 = -0.2 * random()
        sample[0] += d_h_negative1
        sample[1] += d_h_positive
        sample[2] += d_h_negative2

        augmented_category_2.append(sample)

    # augment category 3
    while len(augmented_category_3) < max_elem_per_category - len(category_3):
        sample = choice(category_3)

        # subtract from loose block and add to others
        d_h_positive1 = 0.2 * random()
        d_h_positive2 = 0.2 * random()
        d_h_negative = -0.2 * random()
        sample[0] += d_h_positive1
        sample[1] += d_h_negative
        sample[2] += d_h_positive2

        augmented_category_3.append(sample)

    category_0 = category_0 + augmented_category_0
    category_1 = category_1 + augmented_category_1
    category_2 = category_2 + augmented_category_2
    category_3 = category_3 + augmented_category_3

    x = category_0 + category_1 + category_2 + category_3
    x = np.array(x)

    y = [0] * len(category_0) + [1] * len(category_1) + [2] * len(category_2) + [3] * len(category_3)
    y = np.array(y)

    return x, y

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        acc = logs.get("acc")
        val_acc = logs.get("val_acc")
        if acc == 1. and val_acc == 1:
            self.model.stop_training = True



if __name__ == "__main__":
    # dataset ready!

    x, y = read_data()
    x, y = augment_data(x, y)
    print(f"Augmented data: {x}")
    x, y, mean, std = prepare_data(x, y)

    print(f"x: {x * std + mean}")
    print(f"y: {y}")

    print(f"Data length: {len(x)}")

    num_classes = 4
    input_shape = 3

    model = keras.Sequential()
    model.add(Dense(10, input_dim=3, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # es = EarlyStopping(monitor='loss', mode='auto', patience=10,verbose=1)
    es = EarlyStoppingAtMinLoss()

    model.fit(x=x, y=y, nb_epoch=2000, verbose=1, validation_split=0.33, shuffle=True, callbacks=[es])

    model.save_weights('loose_blocks_model')

    model.load_weights('loose_blocks_model')


    # x, y = read_data()
    # x, y, _, _ = prepare_data(x, y, mean, std)
    print(f"x: {x}")
    print(f"y: {y}")

    results = model.evaluate(x=x, y=y)
    print(f"test loss, test acc: {results}")
    falses = 0
    for i in range(len(x)):
    # for i in range(10):
        print(f"x: {x[i] * std + mean}")
        prediction = model.predict(np.reshape(x[i], (1,3)))

        truth_pos = np.argmax(y[i])
        prediction_pos = np.argmax(prediction)
        if truth_pos != prediction_pos:
            falses += 1
        print(f"Prediction: {prediction}")
        print(f"Truth: {y[i]}")

    print(f"Falses: {falses}")
    print(f"Accuracy = {(len(x) - falses) / len(x)}")


