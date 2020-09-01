import csv
from cv.blocks_calibration import read_block_sizes
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from scipy import stats


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

        print(row)
        dataset.append(row)

dataset = np.reshape(dataset, (len(dataset), 4))

x = dataset[:, :3]
x = stats.zscore(x)
y = dataset[:, 3]
y = to_categorical(y, 4)

print(x)
print(y)

# dataset ready!


num_classes = 4
input_shape = 3

model = keras.Sequential()
model.add(BatchNormalization())
model.add(Dense(10, input_dim=3, activation='tanh'))
model.add(BatchNormalization())
model.add(Dense(10, activation='tanh'))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=x, y=y, nb_epoch=2000, verbose=1)




