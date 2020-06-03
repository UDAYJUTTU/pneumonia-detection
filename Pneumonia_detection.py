-*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:31:50 2019

@author: udayjuttu
"""

from numpy import newaxis
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import zoom
from keras import optimizers
from keras.utils import plot_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Sequential

# reading train label dataset to pandas
df = pd.read_csv('/content/stage_2_train_labels.csv/stage_2_train_labels.csv')

def parse_data(df):

    # Method to read a CSV file (Pandas dataframe) and parse the data into the following nested dictionary
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]
    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '/content/stage_2_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed

patient_id = parsed.keys()
patient_id = list(patient_id)
x_ray = []
label = []
box_parm = []

for i in parsed:
    x_ray.append(parsed[i]['dicom'])
    # print(x_ray)
    label.append(parsed[i]['label'])
    box_parm.append(parsed[i]['boxes'])


patient_df = pd.DataFrame(np.column_stack([patient_id, x_ray, label, box_parm]),
                          columns=['patient_id', 'x_ray', 'label', 'box_parm'])


detail_class_df = pd.read_csv('/content/stage_2_detailed_class_info.csv')

patient_df.columns.values


X_list = []
for i in patient_df['x_ray']:
    data = i
    d = pydicom.read_file(data)
    arr = d.pixel_array
    assert np.min(arr) >= 0 & np.max(arr) <= 255
    size = 256
    resize_factor = float(size) / np.max(arr.shape)
    resized_img = zoom(arr, resize_factor, order=1, prefilter=False)
    l = resized_img.shape[0];
    w = resized_img.shape[1]
    if l != w:
        ldiff = (size - l) / 2
        wdiff = (size - w) / 2
        pad_list = [(ldiff, size - l - ldiff), (wdiff, size - w - wdiff)]
        resized_img = np.pad(resized_img, pad_list, "constant", constant_values=0)
    assert size == resized_img.shape[0] == resized_img.shape[1]
    X_list.append(resized_img.astype("uint8"))


myInt = 225
X_list[:] = [np.float32(x) / myInt for x in X_list]

patient_df['x_ray_array'] = pd.Series(X_list)

X = X_list
Y = patient_df['label']

# split the data into train and test
random_seed = 2
X_t, X_valid, y_t, y_valid = train_test_split(X, Y, test_size=0.2, random_state=random_seed)
X_t1, X_t2, y_t1, y_t2 = train_test_split(X_t, y_t, test_size=0.3, random_state=random_seed)


# del X_list
del patient_df

X_validation = np.array(X_valid)
X_train = np.array(X_t1)
X_test = np.array(X_t2)
y_validation = np.array(y_valid)
y_train = np.array(y_t1)
y_test = np.array(y_t2)
X_validation = X_validation[..., newaxis]
X_train = X_train[..., newaxis]
X_test = X_test[..., newaxis]

## Designing model

lenet = Sequential()

# layer_1
lenet.add(Conv2D(96, kernel_size=11, strides=4, padding='same', input_shape=(256, 256, 1), activation='relu'))
lenet.add(MaxPool2D(pool_size=2, strides=2, padding='same'))
# layer_2
lenet.add(Conv2D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
lenet.add(MaxPool2D(pool_size=2, strides=2, padding='same'))
# layer_3
lenet.add(Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu'))
lenet.add(MaxPool2D(pool_size=2, strides=2))
# layer_4
lenet.add(Conv2D(256, kernel_size=2, strides=1, padding='valid', activation='relu'))
lenet.add(MaxPool2D(pool_size=2, strides=2))
lenet.add(Flatten())
lenet.add(Dense(120))
lenet.add(Dense(84))
lenet.add(Dense(2, activation='softmax'))
lenet.summary()


plot_model(lenet, to_file='lenet.png', show_shapes=True)
RMSprop = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0005)
lenet.compile(optimizer=RMSprop, loss='categorical_crossentropy', metrics=['accuracy'])
lenet.fit(X_train, y_train, batch_size=1000, epochs=20, validation_data=[X_validation, y_validation])
