import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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

categorical_variables = chi_2_features.copy()
categorical_variables.remove("KHscore")
df_categorical = pd.get_dummies(cfpb_data[categorical_variables], drop_first=True)
df_numerical = cfpb_data["KHscore"]
df = pd.concat([df_categorical, df_numerical], axis=1)
X = cfpb_data[mutual_information_features]
y = cfpb_data[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.2, random_state=0)
X1 = sm.add_constant(df)
model = sm.Logit(y_train, X_train)
results = model.fit(maxiter=500)
prob_pred = results.predict(X_test)
y_pred = [0 if x < .25 else 1 for x in prob_pred]

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Predicted prob':prob_pred})

if __name__ == "__main__":
    print(results.summary())
    print(classification_report(df['Actual'], df['Predicted'], digits=3))
