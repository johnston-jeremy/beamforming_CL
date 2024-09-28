from pdb import set_trace as bp
from multiprocessing import Pool, Manager
from time import time
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import numpy.linalg as la


def mmse_beamformers(H, P):
    K,N = H.shape
    W = la.solve(np.eye(N)*K/P + H.T.conj()@H, H.T.conj())
    W = W/la.norm(W,axis=0,keepdims=True)/np.sqrt(K)

    return W

def sum_rate(W,H):
    HW = H@W
    absHW2 = np.abs(HW)**2
    S = np.diagonal(absHW2)
    I = np.sum(absHW2, axis=-1) - S
    N = 1
    SINR = S/(I+N)
    return np.log2(1+SINR).sum()

def slsqp(H, storage, SNRdb):
    K,N = H.shape

    sumrate = []
    Wall = []
    for P in 10**(SNRdb/10):
        consW = {'fun':partial(W_constraint, P=P, N=N, K=K), 'type':'ineq'}

        W = mmse_beamformers(H,P) 
        W0 = W * np.sqrt(P)

        x0 = complex_to_real(W0)
        objective = partial(sum_rate, H=H)
        sol = minimize(lambda x: -objective(real_to_complex(x,N,K)), x0=x0, method='SLSQP', constraints=[consW])
        W = real_to_complex(sol.x,N,K)
        W = np.sqrt(P) * W/la.norm(W)
        sumrate.append(sum_rate(W, H))
        Wall.append(W)

    storage.append({'H':H, 'W':Wall, 'sumrate':sumrate})

def real_to_complex(z, N, K):     
    W = (z[:len(z)//2] + 1j * z[len(z)//2:]).reshape((N,K))
    return W

def complex_to_real(W):
    return np.concatenate((W.real.ravel(), W.imag.ravel()))

def W_constraint(z,P,N,K):
  W = (z[:len(z)//2] + 1j * z[len(z)//2:]).reshape((N,K))
  return P-la.norm(W)**2

if __name__ == '__main__':
    N = K = 16
    num_samples = 10
    SNRdb = np.array([0,5,10,15,20])
    tstart = time()

    # load samples
    Hall = (np.random.randn(num_samples,N,K) + 1j*np.random.randn(num_samples,N,K))/np.sqrt(2)

    ## multiprocessing 
    # manager = Manager()
    # storage_slsqp = manager.list()
    # with Pool() as pool:
    #     pool.map(partial(slsqp, storage=storage_slsqp, SNRdb=SNRdb), Hall)

    storage_slsqp = []
    for H in tqdm(Hall):
        slsqp(H=H, storage=storage_slsqp, SNRdb=SNRdb)

    sumrate_slsqp = np.array([s['sumrate'] for s in storage_slsqp])
    sumrate_slsqp_avg = np.mean(sumrate_slsqp, axis=0)

    sumrate_mmse = []
    for H in Hall:
        W = [(P**0.5)*mmse_beamformers(H, P) for P in 10**(SNRdb/10)]
        sumrate_mmse.append([sum_rate(w,H) for w in W])
    sumrate_mmse_avg = np.mean(sumrate_mmse, axis=0)

    print('SLSQP', sumrate_slsqp_avg.tolist())
    print('MMSE', sumrate_mmse_avg.tolist())
    print(time()-tstart, 'seconds')
    plt.plot(SNRdb, sumrate_slsqp_avg)
    plt.plot(SNRdb, sumrate_mmse_avg)
    plt.legend(['SLSQP', 'MMSE'])
    plt.grid(linestyle='--')
    plt.show()