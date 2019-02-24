from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
 
def load_attributes(path):
    columns = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(path, sep=" ", header=None, names=columns)

    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    for (zipcode, count) in zip(zipcodes, counts):
        if count < 20:
            rows = df[df['zipcode'] == zipcode].index
            df = df.drop(rows)
    
    return df

def process_attributes(df, train, test):
    continuous_attributes = ["bedrooms", "bathrooms", "area"]

    scaler = MinMaxScaler()
    trainContinuous = scaler.fit_transform(train[continuous_attributes])
    testContinuous = scaler.transform(test[continuous_attributes])

    binarizer = LabelBinarizer().fit(df['zipcode'])
    trainCategorical = binarizer.transform(train['zipcode'])
    testCategorical = binarizer.transform(test['zipcode'])

    train = np.hstack([trainContinuous, trainCategorical])
    test = np.hstack([testContinuous, testCategorical])

    return train, test

def load_images(df, path):
    images = []

    for i in df.index.values:
        base_path = os.path.sep.join([path, "{}_".format(i+1)])
        house_paths = []
        for img in os.listdir(path):
            if img.startswith("{}_".format(i+1)):
                house_paths.append(path + "/" + img)
        
        house_paths.sort()

        input_images = []
        output_image = np.zeros((64, 64, 3), dtype="uint8")

        for hp in house_paths:
            img = cv2.imread(hp)
            img = cv2.resize(img, (32, 32))
            input_images.append(img)

        output_image[0:32, 0:32] = input_images[0]
        output_image[0:32, 32:64] = input_images[1]
        output_image[32:64, 32:64] = input_images[2]
        output_image[32:64, 0:32] = input_images[3]

        images.append(output_image)

    return np.array(images)