import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
plt.style.use('ggplot')

def plot_confusion_matrix(Y_true, Y_pred):
    plt.figure(figsize=(2,2))
    cm = confusion_matrix(Y_true, Y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)
    plt.imshow(cm, cmap=plt.cm.coolwarm)
    plt.show()
    
def print_precision_recall_fscore(Y_true, Y_pred):
    metrics = precision_recall_fscore_support(Y_true, Y_pred)
    print('Precision:', metrics[0], metrics[0].mean())
    print('Recall:', metrics[1], metrics[1].mean())
    print('F-score:', metrics[2], metrics[2].mean())
    
def plot_prediction_surface(X_shared, y_shared, X_test, model, trace, pred, dims):
    X = X_shared.get_value()
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    xmean = X.mean(axis=0)

    grid = np.mgrid[xmin[dims[0]]:xmax[dims[0]]:100j,xmin[dims[1]]:xmax[dims[1]]:100j]
    grid_2d = grid.reshape(2, -1).T
    
    concat = []
    for i in range(X.shape[1]):
        if i not in dims:
            concat.append(np.ones((grid_2d.shape[0],1)) * xmean[i])
    concat.append(grid_2d)

    grid_2d = np.concatenate(concat, axis=1)
    dummy_out = np.ones(grid.shape[1], dtype=np.int8)

    X_shared.set_value(grid_2d)
    y_shared.set_value(dummy_out)

    # Creater posterior predictive samples
    with model:
        ppc = pm.sample_ppc(trace, samples=50)

    contour_pred = ppc['out'].mean(axis=0)
    contour_uncert = ppc['out'].std(axis=0)

    # draw surface proability
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.rainbow_r
    cmap = plt.cm.binary_r

    contour = ax[0].contourf(grid[0], grid[1], contour_pred.reshape(100, 100), cmap=cmap)
    ax[0].scatter(X_test[pred==0, dims[0]], X_test[pred==0, dims[1]], c='red')
    ax[0].scatter(X_test[pred==1, dims[0]], X_test[pred==1, dims[1]], c='blue')
    cbar = plt.colorbar(contour, ax=ax[0])
    _ = ax[0].set(xlim=(xmin[dims[0]], xmax[dims[0]]), ylim=(xmin[dims[1]], xmax[dims[1]]), xlabel='X', ylabel='Y');
    cbar.ax.set_ylabel('Posterior predictive mean probability of class');

    # draw surface uncertainity
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    contour = ax[1].contourf(grid[0], grid[1], contour_uncert.reshape(100, 100), cmap=cmap)
    ax[1].scatter(X_test[pred==0, dims[0]], X_test[pred==0, dims[1]], c='red')
    ax[1].scatter(X_test[pred==1, dims[0]], X_test[pred==1, dims[1]], c='blue')
    cbar = plt.colorbar(contour, ax=ax[1])
    _ = ax[1].set(xlim=(xmin[dims[0]], xmax[dims[0]]), ylim=(xmin[dims[1]], xmax[dims[1]]), xlabel='X', ylabel='Y');
    cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)');
