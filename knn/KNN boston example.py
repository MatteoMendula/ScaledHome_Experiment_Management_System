#!/usr/bin/env python
# coding: utf-8

# In[126]:


from sklearn.datasets import *
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


# In[113]:


boston = load_boston()
print(boston["filename"])
boston.keys()


# In[114]:


bos = pd.DataFrame(boston['data'])
bos.columns = boston['feature_names']
bos['Price'] = boston['target']
bos.head()


# In[140]:


def calcolateMSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def predictWithKnnInput(knn, train_x, train_y, to_be_predicted):
    return knn.predict(to_be_predicted)

def instantiateKnn(metricgiven, n_neighbors, weights="uniform", algorithm="auto"):
    knn = KNeighborsRegressor(n_neighbors, metric=metricgiven, weights=weights, algorithm=algorithm)
    return knn

def predictWithBestParameters(train_x, train_y, metrics_list, weights_list, value_to_predict, real_value):
    best = {"prediction": None, "mse": None, "n_neighbors": None, "metric": None, "weights": None}
    for weights in weights_list:
        for n_neighbors in range(1, train_x.shape[0]+1):
            for metric in metrics_list:
                #                 print(f"{metric} {n_neighbors} {weights}")
                knn_i = instantiateKnn(metric, n_neighbors, weights)
                #                 train_x_1 = train_x.copy()
                #                 train_y_1 = train_y.copy()
                #                 print(train_x.shape, train_y.shape)
                knn_i.fit(train_x, train_y)
                prediction = knn_i.predict(value_to_predict)
                mse = calcolateMSE([real_value],[prediction])
                if (best["mse"] == None or mse < best["mse"]):
                    best = {"prediction": prediction, "mse": mse, "n_neighbors": n_neighbors, "metric": metric, "weights": weights}
    return best


# In[133]:


# metrics which need parameters cannot be run with auto -> ball_tree seems problematic
metrics_list = ["chebyshev", "minkowski", "manhattan", "euclidean"]
weights_list = ["uniform", "distance"]


# In[117]:


train_x = boston['data'][:500]
train_y = boston['target'][:500]


            
KNR = KNeighborsRegressor(499, metric="manhattan", weights="uniform", leaf_size=30)

print(KNR)

KNR.fit(train_x, train_y)

# print(train_x)


test = np.array(boston['data'][504])
test1 = test.reshape(1,-1)

print(test1)

# bos.loc[500:]


# In[118]:


KNR.predict(test1)


# In[119]:


calcolateMSE([22],[KNeighborsRegressor(499, metric="manhattan", weights="uniform", leaf_size=30).fit(train_x, train_y)
.predict(test1)])


# In[120]:


calcolateMSE([22],[KNeighborsRegressor(499, metric="mahalanobis", weights="uniform", leaf_size=50).fit(train_x, train_y)
.predict(test1)])


# In[141]:


best_parameters = predictWithBestParameters(train_x, train_y, metrics_list, weights_list, test1, 22)


# In[142]:


best_parameters

