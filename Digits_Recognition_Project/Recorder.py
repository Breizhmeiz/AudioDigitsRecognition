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
import os

# %%
# Pour le dev rec2
from importlib import reload  # Python 3.4+

import Tools.tools
reload(Tools.tools)

# %%
from Tools.tools import rec_m

# %% [markdown]
# # Application en Temps Réel

# %% [markdown]
# ## Import du modèle

# %%
# Avec pickle
from pickle import load
# load the model from disk
filename = 'best_model.pkl'
model = load(open(filename, 'rb'))
model

# %%
# Avec joblib
from joblib import load
# load the model from disk
filename = 'best_model.sav'
model = load(filename)
model

# %% [markdown]
# ## Faire un test

# %%
pred = rec_m(model)

# %%
# Affichage de la prédiction :

print("Prediction : ", round(pred[0]))
for p in pred:
    print(p)

# %%
# Pour Tensorflow :

print("Prediction : ", np.argmax(pred[0]))
for p in pred[0]:
    print(p)

# %%

# %%
