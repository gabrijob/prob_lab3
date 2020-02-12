# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import digamma
import pdb
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import json

### Global variables
K=3 # Number of components
N=100 # Number of observations
Dx=2 # Dimension of the observation space
nIter = 10 # VEM Iterations
do_plot = 1 # In the general case
colors = ['r','g','b']

### Model's parameters (same for all sequences)
## Components weights
GroundTruth_pi_k = (1.0/K)*np.ones((1,K))
## Prior for the means
# Mean of the mean prior
GroundTruth_m = np.zeros((Dx))
# Covariance of the mean prior
GroundTruth_omega = 100
GroundTruth_Omega = GroundTruth_omega*np.identity(Dx)
## Priof of the variance
GroundTruth_alpha = 40
alpha = GroundTruth_alpha # No estimation of the parameter alpha
GroundTruth_beta = .1


### Data Generation
## Assignment Ground Truth
GroundTruth_Assignment = np.random.randint(0,K,N)
## The means
GroundTruth_Means = np.random.multivariate_normal(mean = GroundTruth_m, cov = GroundTruth_Omega, size = K)
## The variances
GroundTruth_Variances = np.random.gamma(shape = GroundTruth_alpha, scale = GroundTruth_beta, size = K)
## Generate the observations
x = np.zeros((Dx,N))
for no in range(N):
    target_index = GroundTruth_Assignment[no];
    x[:,no] = np.random.multivariate_normal(mean = GroundTruth_Means[target_index,:], cov = GroundTruth_Variances[target_index]*np.identity(Dx))
## Plot
if do_plot:
    plt.figure(1)
    plt.plot(x[0,:],x[1,:],'ob')
    for k in range(K):
        plt.plot(x[0,GroundTruth_Assignment==k],x[1,GroundTruth_Assignment==k],'o'+colors[k])
    plt.show()


data = {}
data['x'] = []
data['Assignments'] = []
data['Means'] = []
data['Variances'] = []
for n in range(N):
    data['x'].append({
        'N': n,
        'X1': x[0,n],
        'X2': x[1,n]
    })
    data['Assignments'].append({
        'N': n,
        'K': GroundTruth_Assingment[n]
    })
for k in range(K):
    data['Means'].append({
        'K': k,
        'Mean': GroundTruth_Means[k]
    })
    data['Variances'].append({
        'K': k,
        'Variance': GroundTruth_Variances[k]
    })

with open('distribution.txt', 'w') as outfile:
    json.dump(data, outfile)

### Initialisation
## Means
# Option 1: Select randomly K observations
selected_obs = np.random.choice(range(N),K)
m_k = x[:,selected_obs]
# Option 2: Ground truth initialisation
# m_k = GroundTruth_Means.copy()
# m_k = np.transpose(m_k)

# Initialise Omega_k to identity
Omega_k = np.zeros((Dx,Dx,K))
for k in range(K):
    Omega_k[:,:,k] = np.identity(Dx)
## Assignments
lambda_nk = np.zeros((N,K))
# Compute distance from each observation to the new means
distances = np.zeros((N,K))
for n in range(N):
    for k in range(K):
        distances[n,k] = np.linalg.norm(m_k[:,k]-x[:,n])
    # Closest mean
    kmin = np.argmin(distances[n,:])
    # Hard assignment initialisation
    lambda_nk[n,kmin] = 1
# Plot the hard-assignment
if do_plot:
    plt.figure(2)
    plt.plot(x[0,:],x[1,:],'ob')
    Plot_Assignments = np.argmax(lambda_nk,axis=1)
    for k in range(K):
        plt.plot(x[0,Plot_Assignments==k],x[1,Plot_Assignments==k],'o'+colors[k])
    plt.show()
## Initialise the posterior parameters of the gamma distribution to zero, they are the first to be computed
alpha_k = alpha*np.ones((K))
# Rate parameter
beta_k = np.zeros((K))
eta_k = np.zeros((K))
rho_k = np.zeros((K))
for k in range(K):
    # Average cluster distance
    avg_dist_k = np.average( distances[lambda_nk[:,k]==1,k] )
    # beta = (alpha-1)*average + 1
    # the "+1" avoids singularities
    beta_k[k] = (alpha_k[k]-1)*avg_dist_k + 1
    # Now we can compute eta_k and rho_k
    eta_k[k] = np.log(beta_k[k]) - digamma.digamma(alpha_k[k])[0]
    rho_k[k] = alpha_k[k]/beta_k[k]

### VEM
## Allocate parameters
pi_k = np.zeros((K))
m = np.zeros((Dx))
Omega = np.zeros((Dx,Dx))
beta = 0
## Performance measures lists
mvd_list = []
acc_list = []
#%%# Iterate
for it in range(nIter):

    ## M step
    pi_k = np.sum(lambda_nk, axis=0)/N
    Q_z = 0
    for n in range(N):
        for k in range(K):
            Q_z = Q_z + lambda_nk[n,k] * np.log(pi_k[k])
    #Q_z_array = np.dot(np.log(pi_k), lambda_nk)
    #Q_z_2 = np.sum(Q_z_array)
    
    # update m, omega and beta
    m = 0
    for k in range(K):
        m = np.add(m, m_k[:,k])
        dot = np.dot(np.subtract(m_k[:,k],m), np.subtract(m_k[:,k],m).T)
        Omega = Omega + dot + Omega_k[:,:,k]
    m = m/K
    Omega = Omega/K
    beta = K*alpha / np.sum(rho_k)

    Q_mu = -1/2 * (K*np.log(np.linalg.norm(Omega)))
    for k in range(K):
        Q_mu = Q_mu -1/2 * (np.dot(np.subtract(m_k[:,k],m).T,
                np.dot(np.linalg.inv(Omega),np.subtract(m_k[:,k],m))))
        Q_mu = Q_mu -1/2 * (np.trace(np.dot(np.linalg.inv(Omega), Omega_k[:,:,k])))

    Q_nu = K*alpha*np.log(beta) - beta* np.sum(rho_k)

    Q = Q_z + Q_mu + Q_nu

    ## E-z step
    for n in range (N):
        bias = 0.01
        denominator = 0
        for l in range(K):
            norm = np.linalg.norm(np.subtract(x[:,n], m_k[:,l]))**2
            exp = -1/2 * (Dx*eta_k[l] + rho_k[l] * (norm + np.trace(Omega_k[:,:,l])))
            denominator = denominator + pi_k[l] * np.exp(exp)
        for k in range(K):
            norm = np.linalg.norm(np.subtract(x[:,n], m_k[:,k]))**2
            exp = -1/2 * (Dx*eta_k[k] + rho_k[k] * (norm + np.trace(Omega_k[:,:,k])))
            numerator = pi_k[k] * np.exp(exp) 
            if numerator > 0 and denominator > 0:
                lambda_nk[n,k] = numerator / denominator
            else:
                lambda_nk[n,k] = bias

    ## E-mu step
    for k in range(K):
        Omega_k_inv = np.linalg.inv(Omega) + np.dot(np.identity(Dx),rho_k[k]*np.sum(lambda_nk[:,k])) + np.identity(Dx)
        Omega_k[:,:,k] = np.linalg.inv(Omega_k_inv)
        
        m_k[:,k] = np.dot(np.linalg.inv(Omega), m)
        for n in range(N):
            m_k[:,k] = np.add(m_k[:,k], rho_k[k]*lambda_nk[n,k]*x[:,n])
        m_k[:,k] = np.dot(Omega_k[:,:,k], m_k[:,k])
        
        #m_k[k] = Omega_k[k] * (np.dot(np.linalg.inv(Omega),m)
        #       + rho_k[k]*np.sum(np.dot(lambda_nk[:,k],x)))        

    ## E-nu step
    for k in range(K):
        alpha_k[k] = alpha + Dx/2 * np.sum(lambda_nk[:,k])
        sum_beta = 0
        for n in range (N):
            sum_beta = lambda_nk[n,k] * (np.linalg.norm(np.subtract(x[:,n], m_k[:,k]))**2 + np.trace(Omega_k[:,:,k]))
        beta_k[k] = beta + 1/2 * sum_beta
        eta_k[k] = np.log(beta_k[k]) - digamma.digamma(alpha_k[k])[0]
        rho_k[k] = alpha_k[k] / beta_k[k]
        
    ### Performance plot
    ## Error measures
    mean_vector_distance = 0
    accuracy = 0
    ## Find the optimal cluster assignment
    # Compute the cost of assigning cluster k, to ground-truth cluster l
    cluster_assignment_cost = np.zeros((K,K))
    for k in range(K):
        for l in range(K):
            cluster_assignment_cost[k,l] = np.sum( lambda_nk[GroundTruth_Assignment==l,k])
    # Obtain the ground-truth-to-estiamted cluster assignment
    row_ind, col_ind = linear_sum_assignment(-cluster_assignment_cost)
    ## Compute L2 distance between means
    mvd = 0
    for k in range(K):
        mvd = mvd + np.linalg.norm(m_k[:,row_ind[k]]-GroundTruth_Means[col_ind[k],:])
    mean_vector_distance = mvd
    mvd_list.append(mean_vector_distance)
    ## Compute the clasification error
    # Optimal found assignment
    Estimated_Assignments = np.argmax(lambda_nk,axis=1)
    acc = 0
    for k in range(K):
        acc = acc + np.sum(Estimated_Assignments[GroundTruth_Assignment==col_ind[k]]==row_ind[k])
    accuracy = acc/N*100
    acc_list.append(accuracy)        
    

#%%## Plot performance measures
plt.figure(3)
plt.plot(range(nIter), mvd_list, label='Mean vector distance')
plt.plot(range(nIter), acc_list, label='Accuracy')
plt.legend()
plt.show()

# Plot the hard-assignment
if do_plot:
    plt.figure(4)
    plt.plot(x[0,:],x[1,:],'ob')
    Plot_Assignments = np.argmax(lambda_nk,axis=1)
    for k in range(K):
        plt.plot(x[0,Plot_Assignments==k],x[1,Plot_Assignments==k],'o'+colors[k])
    plt.show()



#%%## Evaluation of the VEM results
## Error measures
mean_vector_distance = 0
accuracy = 0
## Find the optimal cluster assignment
# Compute the cost of assigning cluster k, to ground-truth cluster l
cluster_assignment_cost = np.zeros((K,K))
for k in range(K):
    for l in range(K):
        cluster_assignment_cost[k,l] = np.sum( lambda_nk[GroundTruth_Assignment==l,k])
# Obtain the ground-truth-to-estiamted cluster assignment
row_ind, col_ind = linear_sum_assignment(-cluster_assignment_cost)
## Compute L2 distance between means
mvd = 0
for k in range(K):
    mvd = mvd + np.linalg.norm(m_k[:,row_ind[k]]-GroundTruth_Means[col_ind[k],:])
mean_vector_distance = mvd
## Compute the clasification error
# Optimal found assignment
Estimated_Assignments = np.argmax(lambda_nk,axis=1)
acc = 0
for k in range(K):
    acc = acc + np.sum(Estimated_Assignments[GroundTruth_Assignment==col_ind[k]]==row_ind[k])
accuracy = acc/N*100

