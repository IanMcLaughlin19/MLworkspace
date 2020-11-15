# Standarize features
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
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
model_svc = svm.SVC(probability=True)
model_svc_fit = model_svc.fit(X_train, y_train)

y_pred_proba = model_svc.predict_proba(X_test)[:, 1]
fpr, tpr, t = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
from sklearn.svm import SVC
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
grid.fit(X_train, y_train)
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
grid.fit(X_train, y_train)

"""accuracy_svm = round(model_svc.score(X_train, y_train) * 100, 2)
print("Model Accuracy is: ", accuracy_svm, "%")
print(classification_report(y_test, model_svc.predict(X_test)))"""
"""plt.plot(fpr, tpr, label="AUC="+str(auc))
plt.legend()
plt.show()"""
grid_predictions = grid.predict(X_test)

# print classification report
if __name__ == "__main__":
 print(classification_report(y_test, grid_predictions))


