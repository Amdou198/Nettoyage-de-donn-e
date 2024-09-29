#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# initialisation

# In[5]:


sns.set()


# In[9]:


data = pd.read_csv('bag_of_words.csv')
data.head()


# In[11]:


data.shape


# In[15]:


courses_info = pd.read_csv('courses_info.csv')
courses_info.head()


# In[17]:


courses_info.shape


# Jointure

# In[20]:


cols = ["title", "theme"]


# In[22]:


cols


# In[24]:


courses_info.info


# In[26]:


tmp = courses_info[cols]
tmp.head()


# In[28]:


data


# Jointure

# In[31]:


df = data.merge(tmp, right_on="title", left_on="titre", indicator=True)
df.head()


# In[33]:


df._merge.value_counts()


# separation des donnés
# 
# SPLIT

# Parti themes :

# In[36]:


themes = df.theme_y.values
themes[:10]


# parti Theme

# In[38]:


names = data.titre.values
names[:10]


# pour ne garde que les mots et supprimer les colonnes non important

# In[43]:


cols = ["titre", "theme_y", "_merge", "title_y"]
X = df.drop(columns=cols).values
X[:10]


# Mise en echelle(scalling)

# In[46]:


std_scale = preprocessing.StandardScaler()


# In[48]:


std_scale.fit(X)


# On transforme nos donnée

# In[51]:


X_scaled = std_scale.transform(X)
X_scaled[:10]


# Grace on a la methode Describe on voit que la moyenne est a 0 et l'ecart est a 1

# In[55]:


pd.DataFrame(X_scaled).describe().round(2).iloc[1:3:, : ]


# Les linkage (les liaison)

# In[58]:


Z = linkage(X_scaled, method="ward")
Z[:10]


# le Dendogramme

# In[61]:


fig, ax = plt.subplots(1, 1, figsize=(10,40))

_ = dendrogram(Z, ax=ax, labels=names, orientation = "left")

plt.title("Hierarchical Clustering Dendrogram")
ax.set_xlabel("Distance")
ax.set_ylabel("Cours")
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=15)


# Le clustering

# Analyse de nos clustering

# In[71]:


k=12


# In[73]:


clusters = fcluster(Z, k, criterion='maxclust')
clusters


# In[75]:


crosstab = pd.crosstab(themes, clusters, dropna=False)
crosstab.rename_axis(columns="cluster", index="theme", inplace=True)
crosstab


# In[77]:


fig, ax = plt.subplots(1,1, figsize=(12,6))
ax = sns.heatmap(crosstab, vmin=0.1, vmax=14, annot=True, cmap="Purples")


# Affichage de cours pour chaque cluster

# In[80]:


df = pd.DataFrame({"name" : names, "theme" : themes, "cluster" : clusters})
df.head()


# repartition par clustering de different thémes 

# In[83]:


for i in range(1, 13) : 
    # on fait une selection
    sub_df = df.loc[df.cluster == i]

    # le cluster en question
    print(f"cluster : {i}")

    # on extrait les noms et les themes de chaque ligne
    names_list = sub_df.name.values
    themes_list = sub_df.theme.values

    # on créé une liste de couple nom/theme
    ziped = zip(names_list, themes_list) 
    txt = [f"{n} ({t})" for n, t in ziped]

    # on transforme en str
    txt = " / ".join(txt)
 
    # on print
    print(txt)
    print("\n\n")


# In[ ]:




