Data is Downloaded from Kaggle and Project is set in Google colab

-- Since data capacity is huge we stored it in Google Drive. we have detailed instructions to Download the and import Data into the Google Drive.
   
   - Download_data
   - import_data_google_drive.py 
   
   
-- Requirements:

$ pip install numpy
$ pip install pandas
$ pip install scipy
$ pip install keras

--libaries

import zipfile
from numpy import newaxis
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import zoom
from keras import optimizers
from keras.utils import plot_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Sequential



--Frameworks used:

TensorFlow 
Keras

--Steps:

- Preprocessed the data and removed duplicates,from the data and stored it in google drive.
- Cross-validated the Data and split into Train, Test and Vadlidate groups.
- Design the Lenet model.
- Optimizer RMSprop.
- Categorical Crossentropy loss function.
- Metrics [accuracy, ROC, AUC curve].
- Run for 20 epochs.

