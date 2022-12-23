import pandas as pd  # pandas is a data manipulation library
import numpy as np  # provides numerical arrays and functions to manipulate the arrays efficiently
import matplotlib.pyplot as plt  # data visualization library
from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
#Q4

def truncated_svd(matrix, rank):
    U, s, Vt = svds(matrix, k=rank)
    U = U[:, :rank]
    s = np.diag(s[:rank])
    Vt = Vt[:rank, :]
    svd_matrix = np.dot(np.dot(U, s), Vt)

    return svd_matrix


svd_matrix = truncated_svd(array, 5)


# Q5
def IMPUTESVD(D, IdNA, r):
    for _ in range(20):
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        Y = U[:, 0: r] * np.sqrt(s[0: r])
        X = Vt.T[:, 0: r] * np.sqrt(s[0: r])
        D = np.multiply((1 - Id_NA), D) + np.multiply(Id_NA, Y @ X.T)
    return D


ans = IMPUTESVD(df_D, Id_NA, 15)
mask1 = ans.lt(1)
mask2 = ans.gt(4)
count = (mask1 | mask2).sum().sum()
count
