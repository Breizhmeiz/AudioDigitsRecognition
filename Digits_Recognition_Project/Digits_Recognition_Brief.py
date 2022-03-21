# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reconnaissance des Digits basée sur les enregistrements Audios 

# %% [markdown]
# ### Importer les bibliothéques necessaires 

# %%
from Tools.tools import rec
from Tools.tools import collection
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style("whitegrid", {'axes.grid' : False})
sb.set(font_scale = 2)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 

# %% [markdown]
# ### Collection 

# %%
# collection()

# %% [markdown]
# #### 1. Importer votre DataSet

# %%
mydata = pd.read_csv('./DataSet/'+os.listdir('./DataSet/')[0])
mydata.shape

# %% [markdown]
# ##### Print

# %%
mydata.head()

# %% [markdown]
# #### 2. Checker les variables quantitatives/qualitatives et les valeurs manquantes 

# %%
#mydata.describe()
#mydata.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
mydata.isna().any()

# %% [markdown]
# #### 3. Visualiser les targets

# %%
mydata['Target']

# %% [markdown]
# #### 4. Notre variable target (Y) est 'gender', Récuprer X et y à partir du jeu de données 

# %%
y = mydata['Target']
X = mydata.iloc[:,:-1]
X.shape

# %%
y.shape

# %% [markdown]
# #### 5. Diviser la DataSet en donneés d'apprentissage et de test (20% pour le test)

# %%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=20/100)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# %% [markdown]
# #### 6. Appliquer une normalisation centrée-réduite aux données en utilisant "StandardScaler"

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X.copy())
x_train_scaled = scaler.transform(x_train)

#And we want to apply this transformation to a new dataset called New_data
x_test_scaled = scaler.transform(x_test)

# %%
print(x_train_scaled.mean(), x_train_scaled.std())
print(x_test_scaled.mean(), x_test_scaled.std())

# %% [markdown]
# #### 7. Développer votre meilleur modèle de classification

# %% [markdown]
# ##### SVM

# %%
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
#best_C = np.logspace(1,9,num=9,base=10,dtype='int')  # range() avec pas logarithmique (de 2^1 à 2^9)
#best_C = range(1, 100, 10)
best_C = range(1, 10)
best_C = [i/100000 for i in range(18830, 18840)]
# 138 : 0.6518018018018019
# 100 : 0.6518018018018019
#   1 : 0.6783783783783783
#   2 : 0.6840840840840843
# .19 : 0.6837837837837838
#.1883
print(best_C)
grid = GridSearchCV(LinearSVC(), {'C': best_C, 'dual': [False]})
grid.fit(x_train_scaled, y_train)
print("Best score : ", grid.best_score_)
print("Best param : ", grid.best_params_)

# %%
model = grid.best_estimator_
model.fit(x_train_scaled, y_train)

# %% [markdown]
# ##### Tensorflow

# %%
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(x_train_scaled.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrainement du modèle
history = model.fit(x_train_scaled,
                    y_train,
                    epochs=20,
                    batch_size=128)

# %%
# évaluation du modèle
test_loss, test_acc = model.evaluate(x_test_scaled, y_test)
print('test_acc: ',test_acc)

# Predictions
predictions = model.predict(x_test_scaled)
print("Meilleure predictions pour : ", np.argmax(predictions[0]))

# %% [markdown]
# ##### XGBoost

# %%
import xgboost as xgb
# read in data
dtrain = xgb.DMatrix(x_train)
dtest = xgb.DMatrix(x_test)
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)


# %%
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Optimiser la structure des données
data_dmatrix = xgb.DMatrix(data=x_train,label=y_train)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(x_train,y_train)

preds = xg_reg.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# k-fold Cross Validation
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print(cv_results.head())

print("Meilleur : ", (cv_results["test-rmse-mean"]).tail(1))

# %%

# %% [markdown]
# # Application en Temps Réel

# %%
# del rec
from Tools.tools import rec

# %%
pred = rec(scaler, model)

# %%
# Pour Tensorflow :

print("Prediction : ", np.argmax(pred[0]))
for p in pred[0]:
    print(p)

# %%

# %%
