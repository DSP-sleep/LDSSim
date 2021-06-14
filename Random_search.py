"""
Functions for random parameter search.
"""

import os
import sys
import pickle
import numpy as np

from numba import jit
from time import time
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from scipy.integrate import odeint
from scipy.signal import periodogram

import warnings
warnings.filterwarnings("ignore")
error_message = 'Excess work done on this call (perhaps wrong Dfun type).'

@jit(nopython=True)
def system(S, t, k, K):
    """
    The function of the system for scipy.integrate.odeint.
    
    Parameters
    --------------
    S : array
    Condition of substrates 
    t : array
    A sequence of time points.
    k : array
    Rate constants.
    K: array
    MM constants.
    
    Returns
    ----------
    Sintg : array
    The change of S.
    """
    
    Sintg = np.empty(6)
    Sa_00, Sa_01, Sa_10, Sb_00, Sb_01, Sb_10 = S
    
    E = 20./(1 + Sa_00/K[0] + Sa_00/K[1] + Sa_01/K[2]   + Sa_10/K[3]
                  + Sb_00/K[8] + Sb_00/K[9] + Sb_01/K[10] + Sb_10/K[11])
    F = 20./(1 + Sa_01/K[4]   + Sa_10/K[5]   + (1000.-Sa_00-Sa_01-Sa_10)/K[6]   + (1000.-Sa_00-Sa_01-Sa_10)/K[7]
                  + Sb_01/K[12] + Sb_10/K[13] + (1000.-Sb_00-Sb_01-Sb_10)/K[14] + (1000.-Sb_00-Sb_01-Sb_10)/K[15])
             
    Sintg[0] = - k[0]*E*Sa_00/K[0] - k[1]*E*Sa_00/K[1] + k[4]*F*Sa_01/K[4] + k[5]*F*Sa_10/K[5]
    Sintg[1] = - k[4]*F*Sa_01/K[4] - k[2]*E*Sa_01/K[2] + k[0]*E*Sa_00/K[0] + k[6]*F*(1000.-Sa_00-Sa_01-Sa_10)/K[6]
    Sintg[2] = - k[5]*F*Sa_10/K[5] - k[3]*E*Sa_10/K[3] + k[1]*E*Sa_00/K[1] + k[7]*F*(1000.-Sa_00-Sa_01-Sa_10)/K[7]
    Sintg[3] = - k[8]*E*Sb_00/K[8] - k[9]*E*Sb_00/K[9] + k[12]*F*Sb_01/K[12] + k[13]*F*Sb_10/K[13]
    Sintg[4] = - k[12]*F*Sb_01/K[12] - k[10]*E*Sb_01/K[10] + k[8]*E*Sb_00/K[8] + k[14]*F*(1000.-Sb_00-Sb_01-Sb_10)/K[14]
    Sintg[5] = - k[13]*F*Sb_10/K[13] - k[11]*E*Sb_10/K[11] + k[9]*E*Sb_00/K[9] + k[15]*F*(1000.-Sb_00-Sb_01-Sb_10)/K[15]
             
    return(Sintg)

def generate_paramset():
    """
    Randomly generate 32 parameters that determine the system.
    
    Returns
    ----------
    k : array
    Rate constants.
    K : array
    MM constants.
    """
    
    rk = np.random.rand(16)
    rK = np.random.rand(16)
    
    k = 10**(3*rk)
    K = 10**(5*rK-2)
    
    return(k, K)

@jit
def check_convergence(v, trange, epsilon=1.0):
    """
    Judge if each state of a substrate is convergent.
    
    Parameters
    --------------
    v : array
    A sequence of a state of a substrate.
    trange : int
    The time the integration was done.
    epsilon : scalar
    A threshold for the judge.
    
    Returns
    ----------
    1 if not convergence.
    """
    
    rang = trange//10
    
    # check convergence
    vstd = np.std(v[-rang:])
    
    diffstd = np.std(np.diff(v[-rang:]))
    if diffstd < epsilon:
        return(3)
    elif vstd < epsilon:
        return(0)
    else:
        return(1) # not convergence
    
def find_chaos(result):
    i_iter = result['i_iter']
    chaos_maybe = result['chaos_maybe']
    
    i_iter += 1
    
    S0 = np.asarray([1000., 0., 0., 1000., 0., 0.]) # initial state of the substrates.
    trange = 1000
    dt = 0.02
    fq = int(1/dt)
    judge = np.zeros(6, dtype='int')

    k, K = generate_paramset() # randomly generate a parameter set

    # First integration to quickly exclude convergent result.
    S, info = odeint(func=DvD, y0=S0, t=np.arange(0, trange, dt), args=(k, K), atol=5.0e-4, rtol=5.0e-4, full_output=1)
    if error_message==info['message']:
        pass
    else:
        for col in range(6):
            judge[col] = check_convergence(v=S[:, col], trange=trange*fq)
        if 1 in judge:
            # Second integration with strict error control parameter to exclude convergent result.
            S, info = odeint(func=DvD, y0=S0, t=np.arange(0, trange, dt), args=(k, K), full_output=1)
            if error_message==info['message']:
                pass
            else:
                for col in range(6):
                    judge[col] = check_convergence(v=S[:, col], trange=trange*fq)
                if 1 in judge:
                    trange = 6000
                    # Third integration to exclude oscillatory results
                    S, info = odeint(func=DvD, y0=S[-1, :], t=np.arange(0, trange, dt), args=(k,K), mxstep=10000, atol=1.0e-5, rtol=1.0e-5, full_output=1)
                    if error_message == info['message']:
                        pass
                    else:
                        # judge whether oscillatory or chaotic. 
                        f, Spw = periodogram(S[int(trange*fq/2):], fs=fq, axis=0)
                        maxfq_row = np.argmax(Spw)//Spw.shape[1]
                        maxfq_col = np.argmax(Spw)%Spw.shape[1]
                        maxfq_rate = np.sum(Spw[maxfq_row-2:maxfq_row+3, maxfq_col])/np.sum(Spw[:, maxfq_col])
                        if 0.15 > maxfq_rate:
                            print('hit!')
                            chaos_maybe.append([k, K]) # seems to be chaos but needs visual inspection

    result = {'i_iter':i_iter, 'chaos_maybe':chaos_maybe}
    return(result)


def random_search(args):
    """
    Iterate random parameter generation and classification of chaotic solutions.
    
    Parameters
    --------------
    args : tuple, shape (2)
        i_core : int
            Specify which cpu core is used.
        n_iter : int
            How much iteration is done by each cpu core.
    """
    
    i_core, n_iter = args
    
    now = datetime.now()
    date = '{}_{}_{}_{}_{}_{}_{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second, i_core)
    np.random.seed(int('{}_{}_{}_{}_'.format(i_core, now.day, now.hour, now.minute)+str(now.microsecond)[-4:-2]))
    
    # the path to save the search results.
    filename = './random_{:02}.pickle'.format(i_core)
    
    chaos_maybe = []
    result = {'i_iter':i_iter, 'chaos_maybe':chaos_maybe}

    st = time()
    for _ in tqdm(range(int(n_iter))):
        result = find_chaos(result)
        
        # save the intermediate result every hour.
        if (time()-st)>60*60:
            with open(filename, 'wb') as f:
                pickle.dump(result, f)

            st = time()

    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(result, f)
    
    print(datetime.now(), 'Core{}: {} chaos_maybe found'.format(i_core, len(result['chaos_maybe'])))
    
def multi_random_search(n_cores, n_iter_per_core):
    """
    A function to do random search using multiple cpu cores.
    
    Parameters
    --------------
    n_cores : int
    How many cpu cores to use.
    n_iter_per_core : int
    How many iterations each core does.
    """
    
    args = []
    for i_core in range(n_cores):
        args.append((i_core, n_iter_per_core))
    
    print('Random search: using {} cores to explore chaos.'.format(n_cores))
    with Pool(processes=n_cores) as pool:
        result = pool.map(random_search, args)
        
if __name__=='__main__':
    _, n_cores, n_iter_per_core = sys.argv
    n_cores = int(n_cores)
    n_iter_per_core = int(n_iter_per_core)
    multi_random_search(n_cores, n_iter_per_core)