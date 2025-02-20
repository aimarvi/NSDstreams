import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from utils import *
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from sklearn.utils import check_random_state
from scipy.special import erfc, erfcinv
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

MACHINE_PREC = np.finfo(float).eps

def truncated_normal_sample(m, s, l, random_state=None):
    """
    Return random number from distribution with density
    p(x)=K*exp(-(x-m)^2/s-l'x), x>=0.
    m and l are vectors and s is scalar
    Adapted from randr function at http://mikkelschmidt.dk/code/gibbsnmf.html
    which is Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk
    """
    rs = check_random_state(random_state)
    sqrt_2s = np.sqrt(2 * s)
    ls = l * s
    lsm = ls - m
    A = lsm / sqrt_2s
    a = A > 26
    x = np.zeros(m.shape)
    y = rs.random_sample(m.shape)
    x[a] = -np.log(y[a]) / (lsm[a] / s)
    na = np.logical_not(a)
    R = erfc(abs(A[na]))
    x[na] = erfcinv(y[na] * R - (A[na] < 0) * (2 * y[na] + R - 2)) * sqrt_2s + m[na] - ls[na]
    x[np.isnan(x)] = 0
    x[x < 0] = 0
    x[np.isinf(x)] = 0
    return x.real

def _sample_factors(N, X, A, B, lmda, bases_cols_to_sample, alpha, mu2, transpose=False):
        #N = self.n_components
        C = np.dot(B, B.T)
        D = np.dot(X, B.T)
        denom = np.diag(C) + lmda
        denom[denom == 0] = MACHINE_PREC
        for n in range(N):
            if bases_cols_to_sample[n]:
                notn = list(range(n)) + list(range(n + 1, N))
                an = (D[:, n] - np.dot(A[:, notn], C[notn, n]) - alpha[:, n] * mu2) / denom[n]
                
                rnorm_variance = mu2 / denom[n]
                A[:, n] = truncated_normal_sample(an, rnorm_variance, alpha[:, n],
                                                  random_state=0)

        if transpose:
            return A.T, C.T, D.T
        return A, C, D
    
def get_consensus_response(data): 
    ##  data is num_runs x num_components x num_images 
    nruns, n_components, _ = data.shape
    data = data.reshape(-1, data.shape[-1])

    merged_spectra = pd.DataFrame(data, columns=range(data.shape[-1]), index=['run%d' % i for i in range(n_components * nruns)])        


    density_threshold = 0.4 #0.5
    k = n_components
    local_neighborhood_size = 0.30

    n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0]/k)

    # Rescale topics such to length of 1.
    l2_spectra = (merged_spectra.T/np.sqrt((merged_spectra**2).sum(axis=1))).T


                #   first find the full distance matrix
    topics_dist = euclidean_distances(l2_spectra.values)
    #   partition based on the first n neighbors
    partitioning_order  = np.argpartition(topics_dist, n_neighbors+1)[:, :n_neighbors+1]
    #   find the mean over those n_neighbors (excluding self, which has a distance of 0)
    distance_to_nearest_neighbors = topics_dist[np.arange(topics_dist.shape[0])[:, None], partitioning_order]
    local_density = pd.DataFrame(distance_to_nearest_neighbors.sum(1)/(n_neighbors),
                                 columns=['local_density'],
                                 index=l2_spectra.index)

    density_filter = local_density.iloc[:, 0] < density_threshold
    l2_spectra = l2_spectra.loc[density_filter, :]

    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans_model.fit(l2_spectra)
    kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)

    # Find median usage for each component across cluster
    median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median() 

    # Normalize median spectra to probability distributions.
    median_spectra = (median_spectra.T/median_spectra.sum(1)).T

    # Compute the silhouette score
    #stability = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')
    data_transformed = median_spectra.values.T

    return data_transformed

def get_consensus_weights(data_response, data_weights, consensus_response):
    nruns, n_components, _ = data_response.shape
    
    all_weights = []
    for c in range(n_components): 
        
        final_weights = []
        for r_ in range(nruns):
           

            corrs = []
            for k in range(n_components):
                corrs.append(pearsonr(consensus_response[:,c], data_response[r_,k,:].T)[0])
            corrs = np.asarray(corrs)
            best_ix = np.argmax(corrs)
            
            vox_weights = data_weights[r_, :, best_ix]
            vox_weights /= vox_weights.sum()
            final_weights.append(vox_weights)
            
        final_weights = np.asarray(final_weights).mean(0)
        all_weights.append(final_weights)
        
    all_weights = np.asarray(all_weights)
    return all_weights

def get_consensus_weights_nmf(A, X, M = 100, N = 20, seed = 1):
    m = 0
    I, J = X.shape
    n_sqrt = np.sqrt(X.mean() / N)
    beta_prior_scalar = 1 / n_sqrt
    beta = np.ones((N, J)) * beta_prior_scalar
    B = np.random.RandomState(seed=seed).exponential(scale=1 / beta_prior_scalar,
                                              size=(N, J))
    mu2 = np.var(X - np.dot(A, B))
    lmda = 0
    components_rows_to_sample = [True] * N
    bases_cols_to_sample = [True] * N
    burn_in = M//2
    n_after_burnin = M - burn_in
    B_mean = np.zeros(B.shape)

    while m < M:
        mu2 = np.var(X - np.dot(A.copy(), B))
        #print(m)
        B, E, F = _sample_factors(N, X.T, B.T, A.copy().T, lmda, components_rows_to_sample, beta.T, mu2, transpose=True)
        if m >= burn_in:
            B_mean += B / n_after_burnin

        m += 1
    return B_mean