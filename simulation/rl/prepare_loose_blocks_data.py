import csv
from cv.blocks_calibration import read_block_sizes
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scipy import stats
from collections import Counter
from random import random, choice, sample
import datetime
from constants import *
from utils.utils import Line
from sklearn.model_selection import train_test_split

def to_categorical(y):
    vector_dim = len(np.unique(y))

    new_y = []

    for label in y:
        # print(f"Label: {label}")
        one_hot = np.zeros(vector_dim)
        one_hot[int(label)] = 1.0
        new_y.append(one_hot)

    new_y = np.reshape(new_y, (len(new_y), vector_dim))

    return new_y


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

def add_one_hot(x, y):
    new_x = []
    new_y = []

    for i in range(len(x)):
        if y[i] == 0:
            new_x.append(list(x[i]) + [1, 0, 0])
            new_y.append(0)
            new_x.append(list(x[i]) + [0, 1, 0])
            new_y.append(0)
            new_x.append(list(x[i]) + [0, 0, 1])
            new_y.append(0)
        if y[i] == 1:
            new_x.append(list(x[i]) + [1, 0, 0])
            new_y.append(1)
            new_x.append(list(x[i]) + [0, 1, 0])
            new_y.append(0)
            new_x.append(list(x[i]) + [0, 0, 1])
            new_y.append(0)
        if y[i] == 2:
            new_x.append(list(x[i]) + [1, 0, 0])
            new_y.append(0)
            new_x.append(list(x[i]) + [0, 1, 0])
            new_y.append(1)
            new_x.append(list(x[i]) + [0, 0, 1])
            new_y.append(0)
        if y[i] == 3:
            new_x.append(list(x[i]) + [1, 0, 0])
            new_y.append(1)
            new_x.append(list(x[i]) + [0, 1, 0])
            new_y.append(0)
            new_x.append(list(x[i]) + [0, 0, 1])
            new_y.append(1)

    new_x = np.reshape(new_x, (len(new_x), 6))
    new_y = np.reshape(new_y, (len(new_y), 1))

    return new_x, new_y

def prepare_data(x, y, mean=None, std=None):
    # shuffle
    y = np.reshape(y, (len(y), 1))
    dataset = np.concatenate((x, y), axis=1)
    np.random.shuffle(dataset)

    x = dataset[:, :6]
    y = dataset[:, 6]

    # normalize
    if mean is None or std is None:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)

    x[:, :3] = (x[:, :3] - mean[:3]) / std[:3]

    # convert to one-hot encoding
    # y = to_categorical(y)

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

    max_elem_per_category = 50000  # 50000  заебись

    d_h_const = 0.05
    # augment category 0
    while len(augmented_category_0) < max_elem_per_category - len(category_0):
        sample = np.array(choice(category_0))

        # add an equal amount to each height
        d_h = random() * 1 - 0.5
        sample = sample + d_h
        augmented_category_0.append(sample)

    # augment category 1
    while len(augmented_category_1) < max_elem_per_category - len(category_1):
        sample = np.array(choice(category_1))

        # subtract height from loose block and add equal amount to others
        d_h_positive = d_h_const * random()
        d_h_negative = -d_h_const * random()

        sample[0] += d_h_negative
        sample[1:] += d_h_positive

        for i in range(10):
            common_const = random() * 1 - 0.5
            new_sample = sample + common_const
            augmented_category_1.append(new_sample)

    # augment category 2
    while len(augmented_category_2) < max_elem_per_category - len(category_2):
        sample = np.array(choice(category_2))

        # subtract from loose blocks and add to other
        d_h_positive1 = d_h_const * random()
        d_h_positive2 = d_h_const * random()
        d_h_negative = -d_h_const * random()
        sample[0] += d_h_positive1
        sample[1] += d_h_negative
        sample[2] += d_h_positive2

        for i in range(10):
            common_const = random() * 1 - 0.5
            new_sample = sample + common_const
            augmented_category_2.append(new_sample)

    # augment category 3
    while len(augmented_category_3) < max_elem_per_category - len(category_3):
        sample = np.array(choice(category_3))

        # subtract from loose block and add to others
        d_h_positive = d_h_const * random()
        d_h_negative1 = -d_h_const * random()
        d_h_negative2 = -d_h_const * random()
        sample[0] += d_h_negative1
        sample[1] += d_h_positive
        sample[2] += d_h_negative2

        for i in range(10):
            common_const = random() * 1 - 0.5
            new_sample = sample + common_const
            augmented_category_3.append(new_sample)

    category_0 = category_0 + augmented_category_0
    category_1 = category_1 + augmented_category_1
    category_2 = category_2 + augmented_category_2
    category_3 = category_3 + augmented_category_3

    x = category_0 + category_1 + category_2 + category_3
    x = np.array(x)

    y = [0] * len(category_0) + [1] * len(category_1) + [2] * len(category_2) + [3] * len(category_3)
    y = np.array(y)

    return x, y

def get_category_heuristic(x, mean, std):
    loose_blocks = []

    [height0, height1, height2] = x[:3] * std[:3] + mean[:3]

    block_pos = None

    if x[3] == 1:
        block_pos = 0
    if x[4] == 1:
        block_pos = 1
    if x[5] == 1:
        block_pos = 2

    # define points
    p0 = (0, height0)
    p1 = (block_width_mean, height0)
    p2 = (block_width_mean, height1)
    p3 = (2*block_width_mean, height1)
    p4 = (2*block_width_mean, height2)
    p5 = (3*block_width_mean, height2)

    if height1 - height0 >= loose_block_height_threshold and height1 - height2 >= loose_block_height_threshold:
        if block_pos == 0 or block_pos == 2:
            return 1
    elif height0 > height2:
        line = Line(p1, p5)
        if line.f(p2[0]) - p2[1] >= loose_block_height_threshold and line.f(p3[0]) - p3[1] >= loose_block_height_threshold:
            if block_pos == 1:
                return 1
        elif p2[1] - line.f(p2[0]) >= loose_block_height_threshold or p3[1] - line.f(p3[0]) >= loose_block_height_threshold:
            if block_pos == 2:
                return 1
    else:
        line = Line(p0, p4)
        if line.f(p2[0]) - p2[1] >= loose_block_height_threshold and line.f(p3[0]) - p3[
            1] >= loose_block_height_threshold:
            if block_pos == 1:
                return 1
        elif p2[1] - line.f(p2[0]) >= loose_block_height_threshold or p3[1] - line.f(
                p3[0]) >= loose_block_height_threshold:
            if block_pos == 0:
                return 1

    return 0

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x, y = augment_data(x_train, y_train)
    x, y = add_one_hot(x, y)
    x, y, mean, std = prepare_data(x, y)

    print(f"Data length: {len(x)}")

    num_classes = 4
    input_shape = 6

    model = keras.Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # es = EarlyStopping(monitor='loss', mode='auto', patience=10,verbose=1)
    es = EarlyStoppingAtMinLoss()

    # tensorboard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x=x, y=y, nb_epoch=2, verbose=1, validation_split=0.2, shuffle=True, callbacks=[es])

    model.save_weights('loose_blocks_model')

    model.load_weights('loose_blocks_model')


    x, y = add_one_hot(x_test, y_test)
    x, y, _, _ = prepare_data(x, y, mean, std)

    results = model.evaluate(x=x, y=y)
    print(f"test loss, test acc: {results}")
    loose_but_not_tested = 0
    stuck_but_tested = 0
    predictions = []
    for i in range(len(x)):
    # for i in range(10):
        print(f"x: {x[i] * std + mean}")
        prediction = model.predict(np.reshape(x[i], (1,6)))
        # prediction = get_category_heuristic(x[i], mean, std)

        truth_pos = y[i]
        prediction_pos = 1 if prediction > 0.2 else 0
        predictions.append(prediction_pos)
        if truth_pos == 1 and prediction_pos == 0:
            loose_but_not_tested += 1
        if truth_pos == 0 and prediction_pos == 1:
            stuck_but_tested += 1
        print(f"Prediction: {prediction}")
        print(f"Truth: {y[i]}")

    predictions_counter = Counter(predictions)
    predicted_loose = predictions_counter[1]
    predicted_stuck = predictions_counter[0]

    truth_counter = Counter(y)
    truth_loose = truth_counter[1]
    truth_stuck = truth_counter[0]

    print(f"Loose but not tested: {loose_but_not_tested}")
    print(f"Stuck but tested: {stuck_but_tested}")
    print(f"Founded blocks ratio = {1 - loose_but_not_tested / truth_loose}")
    print(f"Stuck but tested ratio = {stuck_but_tested / predicted_loose}")
    print(f"Blocks not tested ratio = {predicted_stuck/truth_stuck}")




