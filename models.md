```python
import numpy as np
import pandas as pd
from numba import njit
from scipy import linalg
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
from matplotlib import pyplot as plt

# Takes proper Y, meaning Y.T
def Tanimoto(x, y):
    # Take alpha as 1
    numerator = np.dot(x, y.T)
    denumerator = LA.norm(x) ** 2 + LA.norm(y) ** 2 - numerator
    return numerator / denumerator

# Bandwith 0.72 chosen in Laforgue et al.
def Gaussian(x, y):
    return np.e ** (-0.72 * LA.norm(np.array(x) - np.array(y)) ** 2)
        
def cal_kernel(x, y):
    #return np.dot(x, y.T)
    return (Tanimoto(x, y) + Gaussian(x, y)) / 2

def normalize_kernel(y):
    res = np.zeros((len(y), len(y)))
    for i in range(len(res)):
        #print('?', i, end='\r')
        for j in range(len(res)):
            if i == j:
                res[i][j] = 1
            elif i < j:
                #res[i][j] = cal_kernel(y[i], y[j]) / (np.sqrt(cal_kernel(y[i], y[i])) * np.sqrt(cal_kernel(y[j], y[j])))
                res[i][j] = cal_kernel(y[i], y[j])
            else:
                res[i][j] = res[j][i]
    return res

class IOKR_plus:

    def __init__(self):
        pass

    def fit(self, X, Y, L=1., gamma=1., algo='iokr', alg_param=0., n_epochs=10,
            step_size='auto', mu=1e-8, crit=None):

        # Saving parameters
        self.L = L
        self.gamma = gamma
        self.algo = algo
        self.alg_param = alg_param

        # Training
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        self.n_tr = self.X_tr.shape[0]
        #K_x = np.dot(X, X.T)
        #K_x = normalize_kernel(self.X_tr)
        grid = np.ix_(self.X_tr, self.X_tr)
        #K_x = K_x_avg[grid]
        K_x = K_input[grid]

        if self.algo == 'iokr':
            M = K_x + self.n_tr * self.L * np.eye(self.n_tr)
            self.Omega = np.linalg.inv(M)

        elif self.algo in ['e_krr', 'e_svr', 'k_huber']:

            #K_y = self.Y_tr.dot(self.Y_tr.T)
            #K_y = normalize_kernel(self.Y_tr)
            K_y = K_TG_y[grid]

            self.W, self.Omega, self.objs = compute_Omega(
                algo, self.alg_param, K_x, K_y, self.L, n_epochs, step_size,
                mu, crit=crit)

            if self.algo in ['e_krr', 'e_svr']:
                self.sparsity = np.mean(np.linalg.norm(self.W, axis=1) < 1e-7)
            elif self.algo == 'k_huber':
                self.sparsity = np.mean(self.alg_param / (self.L * self.n_tr) -
                                        np.linalg.norm(self.W, axis=1) < 1e-7)

    def estimate_output_embedding(self, X_te):

        #K_x_te_tr = rbf_kernel(X_te, Y=self.X_tr, gamma=self.gamma)
        grid = np.ix_(X_te, self.X_tr)
        K_x_te_tr = K_input[grid]
        Y_pred = K_x_te_tr.dot(self.Omega).dot(self.Y_tr)
        return Y_pred

    def predict_clamp(self, X_te):

        Y_pred = self.estimate_output_embedding(X_te)
        Y_pred[Y_pred > 0.5] = 1
        Y_pred[Y_pred <= 0.5] = 0
        return Y_pred

    def scores(self, X, Y, k=3):

        # Predict
        H = self.estimate_output_embedding(X)
        Y_pred = self.predict_clamp(X)

        # MSE
        n = X.shape[0]
        mse = np.linalg.norm(Y - H) ** 2
        mse /= n

        # Hamming loss
        n, d = Y.shape
        hamming = np.linalg.norm(Y - Y_pred) ** 2
        hamming *= 100 / (n * d)

        # Precision k
        Pk = Y.ravel()[np.argsort(H, axis=1)[:, :k] +
                       k * np.arange(n).reshape(-1, 1)]
        Pk = np.sum(Pk)
        Pk /= (k * n)

        # Sparsity (or saturation)
        try:
            sparsity = self.sparsity
        except AttributeError:
            sparsity = 0.

        return mse, Pk, hamming, sparsity

    def cross_validation_score(self, X, Y, n_folds=3, L=1., gamma=1.,
                               algo='iokr', alg_param=0., n_epochs=10,
                               step_size='auto', mu=1e-8, crit=None):

        res = np.zeros((n_folds, 7))

        for i in range(n_folds):
            print('Fold:', i + 1)

            # Split
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=1. / n_folds, random_state=2 ** i)

            # Fit
            self.fit(
                X_train, Y_train, L=L, gamma=gamma, algo=algo,
                alg_param=alg_param, n_epochs=n_epochs, step_size=step_size,
                mu=mu, crit=crit)

            res[i, ::2] = np.array(self.scores(X_train, Y_train))      # Train
            res[i, 1::2] = np.array(self.scores(X_test, Y_test))[:-1]  # Test
            
            print()
            print('Fold result:', res)

        res_avg = np.mean(res, axis=0)

        return res_avg


@njit
def objective(UtU, UtV, W, beta):
    A = 0.5 * np.trace(UtU.dot(W).dot(W.T))
    B = - np.trace(UtV.dot(W.T))
    C = beta * np.trace(np.sqrt(W.dot(W.T)))
    return A + B + C


@njit
def BST(x, tau):
    norm_x = np.linalg.norm(x)
    if norm_x >= tau:
        return x - x / norm_x * tau
    else:
        return np.zeros_like(x)


@njit
def proj(x, tau):
    norm_x = np.linalg.norm(x)
    if norm_x < tau:
        return x
    else:
        return x * tau / norm_x


#@njit
def compute_W(algo, alg_param, UtU, UtV, L, W_init, n_epochs, step_size,
              crit=None):
    """
    Compute the solution to the dual problem w.r.t. chosen algorithm
    """
    # Initialization
    n_row = W_init.shape[0]
    W = W_init.copy()
    objs = np.zeros(n_epochs)

    # Iterations
    for s in range(n_epochs):
        print(f'Epoch: {s + 1}', end='\r')
        if algo == 'e_krr':
            obj_param = alg_param
            
            for i in range(n_row):
                # Block Coordinate Descent step
                W[i, :] += 1 / UtU[i, i] * (UtV[i, :] - UtU[i, :].dot(W))
                W[i, :] = BST(W[i, :], alg_param / UtU[i, i])
                
                #print(alg_param / UtU[i, i], UtU[i, i])

        elif algo == 'e_svr':
            obj_param = alg_param
            
            for i in range(n_row):
                # Block Coordinate Descent step
                W[i, :] += 1 / UtU[i, i] * (UtV[i, :] - UtU[i, :].dot(W))
                W[i, :] = BST(W[i, :], alg_param / UtU[i, i])
                # Projection step
                W[i, :] = proj(W[i, :], 1. / (L * n_row))
                obj_param = alg_param
                
                #print(alg_param / UtU[i, i], UtU[i, i])

        elif algo == 'k_huber':
            obj_param = 0.
            
            #print(alg_param / (L * n_row))
            
            # Gradient step
            W -= step_size * (UtU.dot(W) - UtV)
            # Projection step
            for i in range(n_row):
                W[i, :] = proj(W[i, :], alg_param / (L * n_row))

        # Objective and stopping criterion
        objs[s] = objective(UtU, UtV, W, obj_param)
        if crit is not None and s > 0 and np.abs((objs[s - 1] - objs[s]) /
                                                 objs[0]) < crit:
            break

    return W, objs


def compute_Omega(algo, alg_param, K_X, K_Y, L, n_epochs, step_size, mu,
                  crit=None):
    """
    Compute optimal coefficients/outputs matrix Omega w.r.t. chosen algorithm
    """
    n = K_X.shape[0]
    if algo in ['e_krr', 'k_huber']:
        UtU = K_X + n * L * np.eye(n)
    elif algo == 'e_svr':
        UtU = K_X + mu * np.eye(n)

    Q, s, Qt = linalg.svd(K_Y + mu * np.eye(n))
    D = np.diag(s)
    UtV = Q.dot(np.sqrt(D))

    if step_size == 'auto':
        step_size = 8. / (np.trace(K_X) + n * L)

    W_init = np.zeros((n, n))
    W, objs = compute_W(algo, alg_param, UtU, UtV, L, W_init, n_epochs,
                        step_size, crit=crit)

    Omega = W.dot((np.diag(1 / np.sqrt(s))).dot(Qt))

    return W, Omega, objs
```

# Get Input Output


```python
import os
def get_kernels():

    kernelpath = "./data/input_kernels/"
    filenames = os.listdir(kernelpath)
    # there is one file, .DS_Store, that needs to be removed
    filenames = np.sort(filenames)[1:]

    n = 4138

    K = np.empty((len(filenames), n, n))

    for find, fname in enumerate(filenames):
        ktmp = np.loadtxt(kernelpath+fname)

        # centering

        H = np.eye(n) - (1/n)*np.outer(np.ones(n), np.ones(n))
        ktmp = np.dot(H, np.dot(ktmp, H))

        # normalization

        diagonal = np.diag(ktmp)
        diagonal = 1/np.sqrt(diagonal)
        D = np.diag(diagonal)
        ktmp = np.dot(D, np.dot(ktmp, D))

        K[find, :, :] = ktmp

    return K
```


```python
# Load
try:
    K = np.load("tmp_K.npy")
except FileNotFoundError:
    K = get_kernels()
    np.save("tmp_K.npy", K)
    
# Take average of K_x
K_x_avg = np.zeros(shape=(4138, 4138))
for i in range(len(K)):
    K_x_avg += K[i]
K_x_avg /= len(K)

# Mapping
eigvals, eigvecs = np.linalg.eigh(K_x_avg)
vec = eigvals**(-0.5)
vec[eigvals <= 0] = 0
X = np.dot(K_x_avg, np.dot(eigvecs, np.dot(np.diag(vec), eigvecs.T)))
```

    /var/folders/yy/s5v5_zgn6tz4kv3q51y1d2680000gn/T/ipykernel_1003/3651403937.py:16: RuntimeWarning: invalid value encountered in power
      vec = eigvals**(-0.5)



```python
print('X initial', X.shape)
```

    X initial (4138, 4138)



```python
np.save('input.npy', X)
```


```python
K_x_map = np.load('input.npy')
```


```python
from scipy.io import loadmat
A = loadmat("data/data_GNPS.mat")
Y = A["fp"].toarray()
```


```python
print('Y initial', Y.shape)
```

    Y initial (2765, 4138)


# Get Input Kernel, Tanimoto-Gaussian Kernel


```python
K_y = normalize_kernel(Y.T)
np.save('K_y.npy', K_y)
```

    ? 4137 4047


```python
K_TG_y = np.load('K_y.npy')
print(K_TG_y.shape)
```

    (4138, 4138)



```python
K_input = rbf_kernel(K_x_map, Y=K_x_map, gamma=1)
```


```python
from os import listdir
from os.path import isfile, join

candidate_path = 'data/candidates'
files = [f for f in listdir('data/candidates')]
```


```python
B = loadmat(f'{candidate_path}/{files[0]}')
```


```python
B['inchi']
```




    array([[array(['InChI=1S/C23H26N4O5/c1-3-14(2)19(21(29)25-20(22(30)31)15-9-5-4-6-10-15)26-23(32)27-13-18(28)24-16-11-7-8-12-17(16)27/h4-12,14,19-20H,3,13H2,1-2H3,(H,24,28)(H,25,29)(H,26,32)(H,30,31)'],
                  dtype='<U182')                                                                                                                                                                             ],
           [array(['InChI=1S/C23H26N4O5/c24-23(31)25-11-4-7-19(22(29)30)26-21(28)14-27-12-10-17-8-9-18(13-20(17)27)32-15-16-5-2-1-3-6-16/h1-3,5-6,8-10,12-13,19H,4,7,11,14-15H2,(H,26,28)(H,29,30)(H3,24,25,31)'],
                  dtype='<U187')                                                                                                                                                                                  ],
           [array(['InChI=1S/C23H26N4O5/c1-24-20(29)23(10-17-5-4-8-32-17,21(30)25(2)22(24)31)14-26-11-15-9-16(13-26)18-6-3-7-19(28)27(18)12-15/h3-8,15-16H,9-14H2,1-2H3'],
                  dtype='<U147')                                                                                                                                          ],
           ...,
           [array(['InChI=1S/C23H26N4O5/c1-5-26(22(29)32-23(2,3)4)14-15-8-6-7-9-19(15)25-21(28)18-13-24-20-11-10-16(27(30)31)12-17(18)20/h6-13,24H,5,14H2,1-4H3,(H,25,28)'],
                  dtype='<U149')                                                                                                                                            ],
           [array(['InChI=1S/C23H26N4O5/c1-12(2)19(28)25-23-24-18-17(20(29)26-23)16(11-27(18)21(30)13(3)4)10-7-14-5-8-15(9-6-14)22(31)32/h5-6,8-9,11-13H,7,10H2,1-4H3,(H,31,32)(H2,24,25,26,28,29)'],
                  dtype='<U174')                                                                                                                                                                     ],
           [array(['InChI=1S/C23H26N4O5/c1-4-27(5-2)20(28)18(13-15-9-7-6-8-10-15)25-22-24-17-12-11-16(26-23(30)31)14(3)19(17)21(29)32-22/h6-12,18,26H,4-5,13H2,1-3H3,(H,24,25)(H,30,31)'],
                  dtype='<U163')                                                                                                                                                          ]],
          dtype=object)



# Squared Loss


```python
def record_squared(Ls, csv):
    path = f'iokr/P.csv'
    
    # If CSV is available
    if csv:
        i_df = pd.read_csv(path)
        i_df = i_df.loc[:, ~i_df.columns.str.contains('^Unnamed')]
        results = list(i_df.values)
        
        # Filter recorded values
        for i in Ls:
            if i in np.array(results).T[0]:
                Ls.remove(i)
        
    # If this is the first run
    else:
        results = []

    # Record results
    iokr = IOKR_plus()
    for l in Ls:
        print('L', l)
        res = list(iokr.cross_validation_score(np.array(range(4138)), Y.T, 
                                               L=l, gamma=1, algo='iokr', 
                                               alg_param=1, n_epochs=1, 
                                               step_size='auto', mu=1e-8, crit=None))
        res.insert(0, l)
        results.append(res)
        print('mse', res[2], '\n')
    
    # Update CSV
    titles = ['L', 'train_mse', 'test_mse', 'train_pk', 'test_pk', 'train_ham', 'test_ham', 'sparsity']
    df = pd.DataFrame(np.array(results), columns=titles)
    df = df.sort_values(by=['L'])
    df.to_csv(path, index=False)
    return df
```


```python
Ls = [1.e-19, 1.e-8, 1.e-7, 1.e-6, 1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1]
print(Ls)
```

    [1e-19, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1]



```python
record_squared(Ls=[2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7], csv=True)
```

# Kappa Huber


```python
def record_robust(algo, P, P_str, Ls, csv):
    path = f'{algo}/P{P_str}.csv'
    
    # If CSV is available
    if csv:
        i_df = pd.read_csv(path)
        i_df = i_df.loc[:, ~i_df.columns.str.contains('^Unnamed')]
        results = list(i_df.values)
        
        print('Available', np.array(results).T[0])
        # Filter recorded values
        for i in Ls:
            if i in np.array(results).T[0]:
                Ls.remove(i)
    # If this is the first run
    else:
        results = []

    print('Post-filtered', Ls)
    # Record results
    iokr = IOKR_plus()
    for l in Ls:
        print('\nL', l)
        res = list(iokr.cross_validation_score(np.array(range(4138)), Y.T, 
                                               L=l, gamma=1, algo=algo, 
                                               alg_param=P, n_epochs=1, 
                                               step_size='auto', mu=1e-8, crit=None))
        res.insert(0, l)
        results.append(res)
        print('mse', res[2], '\n')
    
    # Update CSV
    titles = ['L', 'train_mse', 'test_mse', 'train_pk', 'test_pk', 'train_ham', 'test_ham', 'sparsity']
    df = pd.DataFrame(np.array(results), columns=titles)
    df = df.sort_values(by=['L'])
    df.to_csv(path, index=False)
    return df
```


```python
record_robust(algo='e_svr', P=0.100002758, P_str='0', Ls=[1e-5], csv=False)
```


```python
record_robust(algo='e_svr', P=1e-2, P_str='2', Ls=[0.2, 0.3, 0.4, 0.5, 1], csv=True)
```

    Available [0.0001 0.001  0.01   0.1   ]
    Post-filtered [0.2, 0.3, 0.4, 0.5, 1]
    L 0.2
    L 0.3
    L 0.4
    L 0.5
    L 1





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L</th>
      <th>train_mse</th>
      <th>test_mse</th>
      <th>train_pk</th>
      <th>test_pk</th>
      <th>train_ham</th>
      <th>test_ham</th>
      <th>sparsity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0001</td>
      <td>1.133740e+07</td>
      <td>1.131098e+07</td>
      <td>0.090151</td>
      <td>0.082689</td>
      <td>59.801647</td>
      <td>59.886329</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0010</td>
      <td>1.422970e+06</td>
      <td>1.419449e+06</td>
      <td>0.089446</td>
      <td>0.083736</td>
      <td>45.821078</td>
      <td>45.929109</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0100</td>
      <td>1.774933e+04</td>
      <td>1.771665e+04</td>
      <td>0.089003</td>
      <td>0.085910</td>
      <td>21.104693</td>
      <td>21.187026</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1000</td>
      <td>1.359736e+02</td>
      <td>1.367405e+02</td>
      <td>0.091360</td>
      <td>0.087601</td>
      <td>6.260883</td>
      <td>6.288239</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.2000</td>
      <td>1.302857e+02</td>
      <td>1.298879e+02</td>
      <td>0.090977</td>
      <td>0.087359</td>
      <td>6.613108</td>
      <td>6.579064</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.3000</td>
      <td>1.548520e+02</td>
      <td>1.540082e+02</td>
      <td>0.089184</td>
      <td>0.091626</td>
      <td>8.797636</td>
      <td>8.732107</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.4000</td>
      <td>1.720655e+02</td>
      <td>1.709888e+02</td>
      <td>0.089204</td>
      <td>0.086795</td>
      <td>8.797642</td>
      <td>8.732107</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.5000</td>
      <td>1.839640e+02</td>
      <td>1.827446e+02</td>
      <td>0.090050</td>
      <td>0.085829</td>
      <td>8.797642</td>
      <td>8.732107</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0000</td>
      <td>2.112782e+02</td>
      <td>2.097666e+02</td>
      <td>0.088600</td>
      <td>0.085427</td>
      <td>8.797642</td>
      <td>8.732107</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def find_test_mses(path, Ls):
    res = []
    df = pd.read_csv(path)
    for i in range(df.shape[0]):
        row = list(df.iloc[i])
        if row[0] in Ls:
            res.append(row[2])
    return res
```


```python
y_k_p5 = find_test_mses('k_huber/P5.csv', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
```


```python
y_k_p4 = find_test_mses('k_huber/P4.csv', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
```


```python
y_k_p3 = find_test_mses('k_huber/P3.csv', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
```


```python
y_k_p2 = find_test_mses('k_huber/P2.csv', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
```


```python
plt.plot(range(len(y_k_p5)), np.array(y_k_p5))
plt.plot(range(len(y_k_p4)), np.array(y_k_p4))
plt.plot(range(len(y_k_p3)), np.array(y_k_p3))
plt.plot(range(len(y_k_p2)), np.array(y_k_p2))
plt.ylim(117, 125)
```




    (117.0, 125.0)




    
![png](output_29_1.png)
    



```python

```
