import numpy as np

arr = np.genfromtxt('a1_lin_data/train.csv')
X = arr[:, :-1]
Y = arr[:, -1]
