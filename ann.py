# Standarize features
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
# Import packages
import matplotlib.pyplot as plt
import pandas as pd

import os
import re
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
from datetime import datetime
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
#import keras.backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

cfpb_data = pd.read_csv("CFPB_data_cleaned.csv")
TARGET_COL = "PRODUSE_3"
mutual_information_features = ['fpl',
 'KHscore',
 'LMscore',
 'FINKNOWL2',
 'KHKNOWL2',
 'ENDSMEET',
 'PRODHAVE_4',
 'PRODUSE_6',
 'EARNERS',
 'MATHARDSHIP_1',
 'MATHARDSHIP_2',
 'COLLECT',
 'REJECTED_2',
 'ABSORBSHOCK',
 'BENEFITS_5',
 'ON1correct',
 'CONNECT',
 'HEALTH',
 'PPETHM',
 'PCTLT200FPL']
chi_2_features = ['fpl',
 'FWBscore',
 'FWB2_3',
 'FSscore',
 'IMPUTATION_FLAG',
 'VALUERANGES',
 'KHscore',
 'SAVINGSRANGES',
 'PRODHAVE_4',
 'PRODUSE_5',
 'PRODUSE_6',
 'MATHARDSHIP_1',
 'ABSORBSHOCK',
 'CONNECT',
 'LIFEEXPECT',
 'EMPLOY1_6',
 'PPEDUC',
 'PPETHM',
 'PPINCIMP',
 'PPMARIT']

categorical_variables = mutual_information_features.copy()
categorical_variables.remove("KHscore")
df_categorical = pd.get_dummies(cfpb_data[categorical_variables], drop_first=True)
df_numerical = cfpb_data["KHscore"]
df = pd.concat([df_categorical, df_numerical], axis=1)
X = cfpb_data[mutual_information_features]
y = cfpb_data[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.2, random_state=0)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

"""model = Sequential()
model.add(Dense(12,input_shape=(80,), activation="relu"))
model.add(Dense(2,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer=Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])
ann = model.fit(x = X_train, y = y_train, verbose=0,epochs=100, validation_split=0.3)"""


from utils import plot_loss

if __name__ == "__main__":
  print("x_train ", X_train.shape)
  print("x_test ", X_test.shape)
  print("y_train ", y_train.shape)
  print("y_test ", y_test.shape)