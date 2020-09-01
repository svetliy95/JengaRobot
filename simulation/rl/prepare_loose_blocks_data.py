import csv
from cv.blocks_calibration import read_block_sizes
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


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

        dataset.append(row)

# dataset ready!


num_classes = 4
input_shape = (3,)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense()
    ]
)




