# clone the dataset repository from https://github.com/emanhamed/Houses-dataset

import datasets
import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os

df = datasets.load_attributes("Houses_Dataset/HousesInfo.txt")
images = datasets.load_images(df, "Houses_Dataset")

images = images/255.0

(trainAttrX, testAttrX, trainImagesX, testImagesX) = train_test_split(df, images,
                                                     test_size=0.25, random_state=42)

max_price = trainAttrX["price"].max()
trainY = trainAttrX["price"] / max_price
testY = testAttrX["price"] / max_price

(trainAttrX, testAttrX) = datasets.process_attributes(df, trainAttrX, testAttrX)

mlp = models.build_mlp(trainAttrX.shape[1], False)
cnn = models.build_cnn(64, 64, 3, regress=False)

concatenated = concatenate([mlp.output, cnn.output])

x = Dense(4, activation="relu")(concatenated)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

optimizer = Adam(lr=1e-3, decay=1e-3/200)
model.compile(loss="mean_absolute_percentage_error", optimizer=optimizer)

model.fit(
	[trainAttrX, trainImagesX], trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=200, batch_size=8)

predictions = model.predict([testAttrX, testImagesX])

difference = predictions.flatten() - testY
percent_difference = (difference/testY)*100
abs_percent_difference = np.abs(percent_difference)

mean = np.mean(abs_percent_difference)
std = np.std(abs_percent_difference)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("Avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("Mean: {:.2f}%, std: {:.2f}%".format(mean, std))
