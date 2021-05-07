from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
import numpy as np

grid = np.arange(-1.5,1.5,0.001).reshape(-1,1)
kT = 2.4943389
files = ["smd1/colvar", "smd2/colvar", "smd3/colvar", "smd4/colvar", "smd5/colvar"]
smd = []

krr = KernelRidge(alpha=0.1,kernel='rbf')

# smooth interpolation of each individual realization
for ifile in files:
    X = np.loadtxt(ifile, skiprows=1, usecols=1).reshape(-1,1)
    y = np.loadtxt(ifile, skiprows=1, usecols=2).reshape(-1,1)
    krr.fit(X,y)
    smd.append(krr.predict(grid))

# cumulant formula
allsmd = np.stack(smd)
fes = np.mean(allsmd,axis=0) - np.var(allsmd,axis=0)/(2.0*kT)

# fit an NN to final estimate of FES
X_train, X_test, y_train, y_test = train_test_split(grid, fes, random_state=9531, shuffle=True, test_size=0.1)

nn = MLPRegressor(hidden_layer_sizes=(12), activation='tanh', solver='adam', alpha=1e-5, early_stopping=False, max_iter=10000, random_state=57451, tol=1e-9)
nn.fit(X_train,y_train)
print(nn.score(X_test,y_test))

grid = np.arange(-1.5,1.51,0.01).reshape(-1,1)
preds = nn.predict(grid).reshape(-1,1)

print(nn.coefs_)
print(nn.intercepts_)

np.savetxt('fes.txt', np.vstack((grid.T, preds.T)).T, fmt='%8.2f  %8.3f')
