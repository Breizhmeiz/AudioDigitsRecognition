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
#from Tools.tools import rec
#from Tools.tools import collection
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
sns = sb
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

from sklearn.pipeline import Pipeline

# %% [markdown]
# ### Collection 

# %%
# collection()

# %% [markdown]
# #### 1. Importer votre DataSet

# %%
mydata = pd.read_csv('./Dataset/'+os.listdir('./Dataset/')[0])
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
print(mydata['Target'])

sns.pairplot(mydata, hue='Target', corner=True)

# %%
# Selection des 2 features les plus représentatives

data = mydata[mydata['Target'].isin([0, 1])]
sns.pairplot(data, hue='Target', corner=True)

# %%
# Selection des paires de features les plus représentatives

for fa, fb in [(1, 3), (3, 4), (5, 8), (11, 5)]:
    data = mydata[mydata['Target'].isin([fa, fb])]
    sns.pairplot(data, hue='Target', corner=True)

# %% [markdown]
# L'idée étant de trouver quelles *features* sont les plus représentatives pour distinguer un chiffre d'un autre, la d
# émarche est interressante, mais pour 10 chiffres, on ira vers un réseau de neurones :-) 

# %%
# Heatmap for correlation matrix
fig, ax = plt.subplots(figsize=(24,12))
sns.heatmap(mydata.corr(),annot=True,linewidths=.5,fmt='.1g',cmap= 'coolwarm')

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
#
# - [X] from sklearn.tree import DecisionTreeClassifier
# - [X] from sklearn.ensemble import RandomForestClassifier
# - [X] from sklearn.ensemble import GradientBoostingClassifier
# - [~] from sklearn.svm import SVC
# - [X] from sklearn.neural_network import MLPClassifier 

# %% [markdown]
# ##### DecisionTree

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


dt = DecisionTreeClassifier()

p = Pipeline([
      ('dt', dt),
    ])
params = {
    'dt__criterion': ['gini', 'entropy'],
    'dt__max_depth': [1, 2, 3, 4, 5, 6, 7],
    'dt__min_samples_split': [2, 3, 5, 10, 15, 20]
    }

gs = GridSearchCV(p, params, cv=3).fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.score(x_test, y_test))

# %%
# Meilleur Modèle :
model = gs.best_estimator_
model.fit(x_train_scaled, y_train)

# %% [markdown]
# ##### RandomForest

# %%
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

#best_n = range(1, 100, 10)
best_n = range(75, 85, 1)
# 91} 0.6411069980609906
# 81} 0.6632293319231447
# 77} 0.6794465009695047
# 84} 0.6958399435924555
print(best_n)

params = {'n_estimators': best_n}
gs = GridSearchCV(rf, params, cv=3).fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.score(x_test, y_test))

# %%
from sklearn.pipeline import Pipeline

p = Pipeline([
      ('rf', rf),
    ])
params = {
    'rf__n_estimators': best_n,
    'rf__criterion': ['gini', 'entropy'],
}

gs = GridSearchCV(p, params, cv=3).fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.score(x_test, y_test))

# %%
# Meilleur Modèle :
model = gs.best_estimator_
model.fit(x_train_scaled, y_train)

# %% [markdown]
# ##### GradientBoosting

# %%
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

best_n = range(1, 150, 10)
#best_n = range(50, 70, 1)
# 61} 0.5866384628944121
# 57} 0.5867265996827076
# 64} 0.5974792878547506
print(best_n)

# à éviter sans une bonne machine :-()
params = {
    'loss':['deviance'],
    'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    'min_samples_split': np.linspace(0.1, 0.5, 12),
    'min_samples_leaf': np.linspace(0.1, 0.5, 12),
    'max_depth':[3,5,8],
    'max_features':['log2', 'sqrt'],
    'criterion': ['friedman_mse', 'squared_error', 'absolute_error'],
    'subsample':[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    'n_estimators': best_n
    }

params = {'n_estimators': best_n}

gs = GridSearchCV(gb, params, cv=3).fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.score(x_test, y_test))



# %%
from sklearn.pipeline import Pipeline

p = Pipeline([
      ('gb', gb),
    ])
params = {
    'gb__n_estimators': best_n,
    'gb__criterion': ['friedman_mse', 'squared_error'],
}

gs = GridSearchCV(p, params, cv=3).fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.score(x_test, y_test))

# %%
# Meilleur Modèle :
model = gs.best_estimator_
model.fit(x_train_scaled, y_train)

# %% [markdown]
# ##### SVM

# %%
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
#best_C = np.logspace(1,9,num=9,base=10,dtype='int')  # range() avec pas logarithmique (de 2^1 à 2^10)
#best_C = range(1, 100, 10)
#best_C = range(1, 10)
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
from sklearn.pipeline import Pipeline

svm = SVC()
print(svm.get_params().keys())

p = Pipeline([
      ('svm', svm),
    ])
params = {
    'svm__C': best_C,
    # 'svm__dual': [False],
    
}

gs = GridSearchCV(p, params, cv=3).fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.score(x_test, y_test))

# %%
# Meilleur Modèle :
model = gs.best_estimator_
model.fit(x_train_scaled, y_train)

# %% [markdown]
# ##### MLP

# %%
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

mlp = MLPClassifier()

print(mlp.get_params().keys())

p = Pipeline([
      ('mlp', mlp),
    ])
params = {
    'mlp__solver': ['lbfgs', 'sgd', 'adam'],
    'mlp__max_iter': range(1, 1000, 100),
    'mlp__alpha': 10.0 ** -np.arange(1, 10),
    'mlp__hidden_layer_sizes':np.arange(10, 15),
    'mlp__random_state':[0],
    #'mlp__n_jobs': [-1]
}

gs = GridSearchCV(p, params, cv=3).fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.score(x_test, y_test))

# %%
# Meilleur Modèle :
model = gs.best_estimator_
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

# %%
# Meilleur Modèle :
model = model
model.fit(x_train_scaled, y_train)

# %% [markdown]
# ##### XGBoost

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
params = {
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10,
}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print(cv_results.head())

print("Meilleur : ", (cv_results["test-rmse-mean"]).tail(1))

# %%
# Meilleur Modèle :
model = xg_reg
#model = cv_results.best_estimator_
model.fit(x_train_scaled, y_train)

# %%
# save in JSON format
model.save_model("model_XGBoot.json")

# %% [markdown]
# #### Pipeline de selection

# %%
p = Pipeline([
    ('dt', dt),
    ('rf', rf),
    ('gb', gb),
    ('svm', svm),
    #('mlp', mlp),
])

best_n = range(75, 85, 1)

params = {
    'dt__criterion': 'entropy', 'dt__max_depth': 5, 'dt__min_samples_split': 5,
    'rf__criterion': 'gini', 'rf__n_estimators': 77,
    'gb__criterion': 'friedman_mse', 'gb__n_estimators': 71,
    'svm__C': 0.1883,
    #'mlp__solver': ['lbfgs', 'sgd', 'adam'],
    #'mlp__max_iter': range(1, 1000, 100),
    #'mlp__alpha': 10.0 ** -np.arange(1, 10),
    #'mlp__hidden_layer_sizes':np.arange(10, 15),
    #'mlp__random_state':[0],
    #'mlp__n_jobs': [-1]

}


gs = GridSearchCV(p, params, cv=3).fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.score(x_test, y_test))

# %%
# Meilleur Modèle :
model = gs.best_estimator_
model.fit(x_train_scaled, y_train)

# %% [markdown]
# # Application en Temps Réel
#
# Voir le notebook `Recorder` !

# %%
# Pour le dev de rec et rec2
from importlib import reload  # Python 3.4+

import Tools.tools
reload(Tools.tools)

# %%
from Tools.tools import rec

# %%
pred = rec(scaler, model)

# %%
# Pour Tensorflow :

print("Prediction : ", np.argmax(pred[0]))
for p in pred[0]:
    print(p)

# %% [markdown]
# ## Enregistrement du modèle

# %%
# Avec pickle
from pickle import dump
# save the model to disk
filename = 'best_model.pkl'
dump(model, open(filename, 'wb'))

# %%
# Avec joblib
from joblib import dump 
# save the model to disk
filename = 'best.sav'
dump(model, filename)

# %%
