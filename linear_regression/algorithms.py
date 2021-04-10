import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import special_ortho_group


def sample_task(p, beta):

    # Sample random orthogonal matrix
    if p > 1:
        U = special_ortho_group.rvs(p)
    elif p == 1:
        U = np.ones((1,1))
    else:
        raise TypeError("Data dimension not valid")

    # Sample eigenvalues and model
    Theta = np.random.uniform(0.0, beta, p)
    Lambda = np.diag(Theta)
    Q = np.matmul(U, np.matmul(Lambda, np.transpose(U)))

    # Sample error variance sigma_gamma^2
    sigma = np.random.uniform(0.0, beta)

    return Q, Theta, sigma

def expected_loss(p, beta, mc=1000):

    '''
    Returns a function such that,
    given theta, computes a Monte Carlo estimate of the expected MSE
    on the task distribution described in sample_task
    ''' 

    EQ = np.zeros((p,p))
    EQT = np.zeros(p)
    ETQT = 0
    Es = 0

    for l in range(mc):
        Q, Theta, sigma = sample_task(p, beta)
        EQ += Q/mc
        EQT += np.matmul(Q, Theta)/mc
        ETQT += np.dot(Theta, np.matmul(Q, Theta))/mc
        Es += sigma/mc

    def compute_expected_loss(thetahat):
        return np.dot(thetahat, np.matmul(EQ, thetahat))/2-np.dot(EQT, thetahat)+ETQT/2+Es/2

    return compute_expected_loss

def variance_hessian(p, Q, mc=1000):

    # For x ~ N(0, Q), compute the expectation of xx^T Q xx^T - Q^3

    V = np.zeros((p,p))
    Q3 = np.linalg.matrix_power(Q, 3)
    for j in range(mc):
        x = np.random.multivariate_normal(np.zeros(p), Q)
        gx = np.outer(x, x)
        V += (np.matmul(gx, np.matmul(Q, gx))-Q3)/mc

    return V

def expected_loss_updated(p, beta, mc=1000):

    '''
    Returns a function such that,
    given theta, alpha, N, computes a Monte Carlo estimate of the expected MSE after an update using SGD
    with N observations, on the task distribution described in sample_task
    '''

    EQ = np.zeros((p,p))
    EQT = np.zeros(p)
    ETQT = 0
    EQ2 = np.zeros((p,p))
    EQ2T = np.zeros(p)
    ETQ2T = 0
    EQ3 = np.zeros((p,p))
    EQ3T = np.zeros(p)
    ETQ3T = 0
    EV = np.zeros((p,p))
    EVT = np.zeros(p)
    ETVT = 0
    Es = 0
    EsQ2 = 0
    for j in range(mc):
        Q, Theta, sigma = sample_task(p, beta)
        EQ += Q/mc
        EQT += np.matmul(Q, Theta)/mc
        ETQT += np.dot(Theta, np.matmul(Q, Theta))/mc
        Q2 = np.linalg.matrix_power(Q, 2)
        EQ2 += Q2/mc
        EQ2T += np.matmul(Q2, Theta)/mc
        ETQ2T += np.dot(Theta, np.matmul(Q2, Theta))/mc
        Q3 = np.linalg.matrix_power(Q, 3)
        EQ3 += Q3/mc
        EQ3T += np.matmul(Q3, Theta)/mc
        ETQ3T += np.dot(Theta, np.matmul(Q3, Theta))/mc
        V = variance_hessian(p, Q)
        EV += V/mc
        EVT += np.matmul(V, Theta)/mc
        ETVT += np.dot(Theta, np.matmul(V, Theta))/mc
        Es += sigma/mc
        EsQ2 += (sigma*np.trace(Q2))/mc

    def compute_expected_loss_updated(thetahat, N, alpha):
        Qterm = np.dot(thetahat, np.matmul(EQ, thetahat))/2-np.dot(EQT, thetahat)+ETQT/2
        Q2term = np.dot(thetahat, np.matmul(EQ2, thetahat))/2-np.dot(EQ2T, thetahat)+ETQ2T/2
        Q3term = np.dot(thetahat, np.matmul(EQ3, thetahat))/2-np.dot(EQ3T, thetahat)+ETQ3T/2
        Vterm = np.dot(thetahat, np.matmul(EV, thetahat))/2-np.dot(EVT, thetahat)+ETVT/2
        return Qterm-2*alpha*Q2term+alpha**2*Q3term+alpha**2*Vterm/N+Es/2+alpha**2*EsQ2/(2*N)

    return compute_expected_loss_updated

def sample_data(M, D, p, beta):

    # Sample M tasks according to the distribution described in sample_task
    # For each task, sample D data points where the features are ~ N(0, Q) and the errors ~ N(0, sigma)

    N = D//2
    X1 = dict()
    Y1 = dict()
    X2 = dict()
    Y2 = dict()
    for t in range(M):
        Q, Theta, sigma = sample_task(p, beta)
        X1[t] = np.random.multivariate_normal(np.zeros(p), Q, N) # dimension N x p
        Y1[t] = np.matmul(X1[t], Theta)+np.random.normal(0, np.sqrt(sigma), N)
        X2[t] = np.random.multivariate_normal(np.zeros(p), Q, N)
        Y2[t] = np.matmul(X2[t], Theta)+np.random.normal(0, np.sqrt(sigma), N)

    return X1, Y1, X2, Y2

def dr_estimate(M, D, p, X1, Y1, X2, Y2):

    W = np.zeros((p,p))
    z = np.zeros(p)
    for t in range(M):
        W += (np.matmul(np.transpose(X1[t]), X1[t]) + np.matmul(np.transpose(X2[t]), X2[t]))/(M*D)
        # Note that we divide by M*D to avoid overflow/stability issues when solving the system of equations
        z += (np.matmul(np.transpose(X1[t]), Y1[t]) + np.matmul(np.transpose(X2[t]), Y2[t]))/(M*D)

    return np.linalg.lstsq(W, z)[0]

def maml_estimate(alpha, M, D, p, X1, Y1, X2, Y2):

    N = D//2

    W = np.zeros((p,p))
    z = np.zeros(p)
    for t in range(M):
        shrinkage = np.eye(p)-(alpha/N)*np.matmul(np.transpose(X1[t]), X1[t])
        W += np.matmul(shrinkage, np.matmul(np.matmul(np.transpose(X2[t]), X2[t]), shrinkage))/(M*N)
        z += np.matmul(shrinkage, np.matmul(np.transpose(X2[t]), Y2[t]-(alpha/N)*np.matmul(X2[t], np.matmul(np.transpose(X1[t]), Y1[t]))))/(M*N)

    return np.linalg.lstsq(W, z)[0]

def compare_mses(p, beta, fig_path):

    num_tasks = [2, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400] # M
    num_data = [1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400] # N
    alphas = np.arange(0.0, 1.05, 0.05)
    trials = 1000

    # Construct functions to compute expected loss before and after a test update
    expected_loss_func = expected_loss(p, beta)
    expected_loss_update_func = expected_loss_updated(p, beta)

    drmse = np.zeros((len(num_tasks), len(num_data)))
    drmse_updated = np.zeros((len(num_tasks), len(num_data), len(alphas)))
    mamlmse = np.zeros((len(num_tasks), len(num_data), len(alphas)))
    mamlmse_updated = np.zeros((len(num_tasks), len(num_data), len(alphas)))
    # Record if MAML estimate has lower loss than DRS
    compare_preupdate = np.zeros((len(num_tasks), len(num_data), len(alphas)))
    compare_postupdate = np.zeros((len(num_tasks), len(num_data), len(alphas)))

    for d in range(len(num_tasks)):
        for e in range(len(num_data)):
            print((num_tasks[d], num_data[e]))
            for l in range(trials):
                X1, Y1, X2, Y2 = sample_data(num_tasks[d], 2*num_data[e], p, beta)
                drest = dr_estimate(num_tasks[d], 2*num_data[e], p, X1, Y1, X2, Y2)
                drmse_pre = expected_loss_func(drest)
                drmse[d, e] += drmse_pre/trials
                for f in range(len(alphas)):
                    drmse_post = expected_loss_update_func(drest, num_data[e], alphas[f])
                    drmse_updated[d,e,f] += drmse_post/trials
                    mamlest = maml_estimate(alphas[f], num_tasks[d], 2*num_data[e], p, X1, Y1, X2, Y2)
                    mamlmse_pre = expected_loss_func(mamlest)
                    mamlmse[d,e,f] += mamlmse_pre/trials
                    mamlmse_post = expected_loss_update_func(mamlest, num_data[e], alphas[f])
                    mamlmse_updated[d,e,f] += mamlmse_post/trials
                    compare_preupdate[d,e,f] += float(drmse_pre > mamlmse_pre)/trials
                    compare_postupdate[d,e,f] += float(drmse_post > mamlmse_post)/trials

    # For each value of alpha, create heatmaps of whether MAML has lower loss than DRS
    # pre and post update
    for f in range(len(alphas)):
        fig = plt.figure(figsize=(3,3))
        im1 = plt.contourf(num_data, num_tasks, compare_preupdate[:,:,f], np.linspace(-0.01, 1.01, 40), vmin=0,vmax=1, cmap='RdBu_r')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("N")
        plt.ylabel("M")
        plt.tight_layout()
        fig.savefig(fig_path+"preupdate_alpha_{}.png".format(int(alphas[f]*100)))
        plt.close(fig)

        fig = plt.figure(figsize=(3,3))
        im2 = plt.contourf(num_data, num_tasks, compare_postupdate[:,:,f], np.linspace(-0.01, 1.01, 40), vmin=0,vmax=1, cmap='RdBu_r')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("N")
        plt.ylabel("M")
        plt.tight_layout()
        fig.savefig(fig_path+"postupdate_alpha_{}.png".format(int(alphas[f]*100)))
        plt.close(fig)
