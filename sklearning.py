# MLP Regression 10-fold CV example using Boston dataset.
import numpy as np
import matplotlib.pyplot as pl
from sklearn import neural_network
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_predict
# ###########################################
# Load data
boston = datasets.load_boston()
# Creating Regression Design Matrix
x = boston.data
# Creating target dataset
y = boston.target
#x_train, x_test, y_train, y_test= cross_validation.train_test_split(x, y, test_size=0.2, random_state=42)
# ######################################################################
# Fit regression model
n_fig=0
for name, nn_unit in [
        ('MLP using ReLU', neural_network.MLPRegressor(activation='relu', solver='lbfgs')), ('MLP using Logistic Neurons', neural_network.MLPRegressor(activation='logistic')), ('MLP using TanH Neurons', neural_network.MLPRegressor(activation='tanh',solver='lbfgs'))
        ]:
    regressormodel=nn_unit.fit(x,y)
    # Y predicted values
    yp =nn_unit.predict(x)
    rmse =np.sqrt(mean_squared_error(y,yp))
    #Calculation 10-Fold CV
    print('Method: %s' %name)
    print('RMSE on the data: %.4f' %rmse)
    n_fig=n_fig+1
    pl.figure(n_fig)
    pl.plot(yp, y,'ro')
    pl.xlabel('predicted')
    pl.title('Method: %s' %name)
    pl.ylabel('real')
    pl.grid(True)
    pl.show()


import numpy as np
import time
import random

xj_ll=np.zeros(2500)
x_sim=np.zeros([2500,2000])
xj_ll=xj_ll+1.1
x_sim=x_sim+1.2
B = 100
def calculate_B(B,xj_ll,x_sim):

    diff2 = []
    for m in range(2 * B):
        mu = np.random.normal(0, 1, 1)
        if (m % 2 == 0):
            diff2.append(np.sin(mu * xj_ll) - np.mean(np.sin(mu * x_sim), axis=1))
        else:
            diff2.append(np.cos(mu * xj_ll) - np.mean(np.cos(mu * x_sim), axis=1))

    return diff2

start = time.time()
diff2=calculate_B(B,xj_ll,x_sim)
end = time.time()

run_time = end - start

print("time = {}".format(run_time))


import numpy as np
import time
import random
import numba
from numba import jit

xj_ll=np.zeros(2500)
x_sim=np.zeros([2500,2000])
xj_ll=xj_ll+1.1
x_sim=x_sim+1.2
B = 100
arr=xj_ll
@jit(nopython=True,parallel = True,fastmath = True)
def calculate_B(arr):

    for m in range(2 * B):
        mu = np.random.normal(0, 1, 1)
        zz=np.sin(mu * x_sim)
        res=0
        for k in range(x_sim.shape[0]):
            for l in range(x_sim.shape[1]):
                res=res+zz[k,l]

#        np.sum(np.sin(mu * x_sim), axis=1)
#          arr=np.concatenate((arr,xj_ll), axis=1)
#         mu = np.random.normal(0, 1, 1)
#         if (m % 2 == 0):
#             np.sin(mu * x_sim)
# #            diff2.append(np.sin(mu * xj_ll) - np.mean(np.sin(mu * x_sim), axis=1))
#         else:
#             np.cos(mu * x_sim)
#            diff2.append(np.cos(mu * xj_ll) - np.mean(np.cos(mu * x_sim), axis=1))

    return arr

start = time.time()
diff2=calculate_B(arr)
end = time.time()

run_time = end - start

print("time = {}".format(run_time))

@jit(nopython=True)
def insertion_sort(arr):

    for i in range(len(arr)):
        cursor = arr[i]
        pos = i

        while pos > 0 and arr[pos - 1] > cursor:
            # Swap the number down the list
            arr[pos] = arr[pos - 1]
            pos = pos - 1
        # Break and do the final swap
        arr[pos] = cursor

    return arr

start = time.time()
list_of_numbers = list()
for i in range(len_of_list):
    num = random.randint(0, len_of_list)
    list_of_numbers.append(num)

for i in range(num_loops):
    result = insertion_sort(list_of_numbers)

import numpy as np
import time
import random
import numba
from numba import jit
xj_ll=np.zeros(2500)
x_sim=np.zeros([2500,2000])
xj_ll=xj_ll+1.1
x_sim=x_sim+1.2
B = 100

start = time.time()
for m in range(2 * B):
    mu = np.random.normal(0, 1, 1)
    np.sin(mu * x_sim)
end = time.time()

run_time = end - start

print("time = {}".format(run_time))

@jit(nopython=True,parallel = True,fastmath = True)
def calculate_B():
    mu = np.random.normal(0, 1, 1)
    result=np.sin(mu * x_sim)
    return result


start = time.time()
for m in range(2 * B):
    calculate_B()
end = time.time()

run_time = end - start

print("time = {}".format(run_time))