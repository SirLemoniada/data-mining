import pandas as pd  # pandas is a data manipulation library
import numpy as np  # provides numerical arrays and functions to manipulate the arrays efficiently
import matplotlib.pyplot as plt  # data visualization library
from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold

df = pd.read_csv("METABRIC_RNA_Mutation.csv")  # Adapt the path
df_D = pd.concat([df['age_at_diagnosis'], df.iloc[:, 31:520]], axis=1)
D = df_D.to_numpy()
y = df['overall_survival_months'].to_numpy()
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv', sep=',')
μ = np.mean(pd.unique(ratings["rating"]))

df_D_NA = ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
)

new = df_D_NA.fillna(μ)
array = new.values
Id_NA = new.isna().to_numpy()

# Q1

ftr = df_D
trgt = pd.DataFrame(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(trgt):
    # Split the data into training and testing sets for this fold
    X_train, X_test = trgt.iloc[train_index], trgt.iloc[test_index]
    # Perform any other operations you need to do on the training and testing sets
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(ftr):
    # Split the data into training and testing sets for this fold
    y_train, y_test = ftr.iloc[train_index], ftr.iloc[test_index]
    # Perform any other operations you need to do on the training and testing sets


def fit_ridge_regression(X_train, y_train, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def predict_ridge_regression(model, X):
    return model.predict(X)


def fit_lasso(X_train, y_train, alpha):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def predict_lasso(model, X):
    return model.predict(X)


# Fit and predict using Ridge
alphas = np.logspace(-10, 8, num=100)
mse_train = []
mse_test = []
selected_features = []
for alpha in alphas:
    model = fit_ridge_regression(X_train, y_train, alpha)
    mse_train.append(np.mean(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')))
    mse_test.append(np.mean(-cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')))
    selected_features.append(np.sum(np.abs(model.coef_) > 1e-16))

# Plot MSE and number of selected features
plt.plot(alphas, mse_train, label='Train')
plt.plot(alphas, mse_test, label='Test')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.plot(alphas, selected_features)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Number of selected features')
plt.show()


# Q4

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


data15 = IMPUTESVD(df_D, Id_NA, 15)

a = data15.loc["1", "1556"]
b1 = data15.loc["91", "2858"]
b2 = data15.loc["91", "1732"]

column_means = data15.mean()
column_means_df = pd.DataFrame(column_means)
c = column_means_df.loc["5313"]

data5 = IMPUTESVD(df_D, Id_NA, 5)
data20 = IMPUTESVD(df_D, Id_NA, 20)
data30 = IMPUTESVD(df_D, Id_NA, 30)

mask5 = (data5 <= 1) | (data5 >= 4)
mask15 = (data15 <= 1) | (data15 >= 4)
mask20 = (data20 <= 1) | (data20 >= 4)
mask30 = (data30 <= 1) | (data30 >= 4)

mask15 = data5.lt(1)
mask25 = data5.gt(4)

mask115 = data15.lt(1)
mask215 = data15.gt(4)

mask120 = data20.lt(1)
mask220 = data20.gt(4)

mask130 = data30.lt(1)
mask230 = data30.gt(4)

count5 = (mask15 | mask25).sum().sum()
count15 = (mask115 | mask215).sum().sum()
count20 = (mask120 | mask220).sum().sum()
count30 = (mask130 | mask230).sum().sum()
