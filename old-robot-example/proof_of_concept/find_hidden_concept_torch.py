"""
limitation: MAP only, can't get full posterior
"""
import numpy as np
import pandas as pd
from torch import nn
from skorch import NeuralNet
import matplotlib.pyplot as plt


def generate_data(N, T, random_state=None):
    """
    """
    np.random.seed(random_state)
    X1 = np.random.choice([-1,0,1], N).astype(float)
    X1n = X1+np.random.randn(N)*0.05
    Z = np.cumsum(X1n)
    X2 = 100-Z+np.random.randn(N)*0.1

    start_ids = np.arange(0, N-T+1, 1)
    X1 = np.array([X1[x:x+T] for x in start_ids])
    Z  = np.array([Z[x:x+T] for x in start_ids])
    X2 = np.array([X2[x:x+T] for x in start_ids])
    df = pd.DataFrame(data=np.c_[X1,Z,X2], columns=[f'X1_t-{T-1-t}' for t in range(T)]+[f'Z_t-{T-1-t}' for t in range(T)]+[f'X2_t-{T-1-t}' for t in range(T)])
    return df


def plot(df, save_path=None):
    """
    """
    import matplotlib.pyplot as plt
    import seaborn
    seaborn.set_style('ticks')
    plt.close()
    fig = plt.figure(figsize=(12,4))

    ax = fig.add_subplot(131)
    ax.scatter(df.X1, df.X2, s=5, c='k')
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    seaborn.despine()

    ax = fig.add_subplot(132)
    xlim = [df.Z.min(), df.Z.max()]
    ax.plot(xlim, xlim, c='r', ls='--')
    ax.scatter(df.Z[:-1], df.Z[1:], s=5, c='k')
    ax.set_xlabel(r'$Z_t$')
    ax.set_ylabel(r'$Z_{t+1}$')
    seaborn.despine()
    
    ax = fig.add_subplot(133)
    ax.scatter(df.Z, df.X2, s=5, c='k')
    #xlim = [min(df.Z.min(), df.X2.min()), max(df.Z.max(), df.X2.max())]
    #ax.plot(xlim, xlim, c='r', ls='--')
    ax.set_xlabel('Z')
    ax.set_ylabel(r'$X_2$')
    seaborn.despine()

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



class Model1:
class L1RegularizedNet(NeuralNet):
    def __init__(self, *args, lambda1=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda1 = lambda1

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
        return loss


#class Model2:


def main():
    random_state = 2022
    N = 100
    T = 3
    df = generate_data(N, T, random_state=random_state)
    print(df.head())
    #plot(df, f'simulated_dataset_N{N}.png')
    X1cols = [x for x in df.columns if x.startswith('X1_')]
    X2cols = [x for x in df.columns if x.startswith('X2_')]
    Zcols  = [x for x in df.columns if x.startswith('Z_')]

    model1 = L1RegularizedNet(
            Model1, module__n_layer=10,
            max_epochs=10,
            lr=0.001, iterator_train__shuffle=True)
    import pdb;pdb.set_trace()
    model1 = GridSearchCV
    model1.fit(df.X1.values, df.X2.values)

    model2 = Model2().fit(df.X1.values, df.X2.values)

    print(f'model1: AIC = {model1.AIC}, BIC = {model1.BIC}')
    print(f'model2: AIC = {model2.AIC}, BIC = {model2.BIC}')

if __name__=='__main__':
    main()
