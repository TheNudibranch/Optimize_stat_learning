import numpy as np
from scipy.stats import multivariate_normal

# Gaussian mixed model for "sequential" data
class Gaus_HMM():
    def __init__(self, clust_num, data=None, tied=False, mus=None, sigmas=None, transitions=None, initials=None):
        if mus is not None:
            self.mus = mus
            self.sigmas = sigmas
            self.transitions = transitions
            self.initials = initials
            self.cluster_num = len(initials)
            self.tied = True if (len(self.sigmas.shape) < 3) else False
        else:
            self.tied = tied
            self.mus = data[np.random.randint(0,len(data), clust_num)]
            c = np.cov(np.vstack((data[:,0], data[:,1])))
            if not self.tied:
                self.sigmas = np.array([np.random.uniform(-np.sqrt(np.abs(c[1,0])),np.sqrt(np.abs(c[1,0])), size=(2,2)) + c for i in range(clust_num)])
            else:
                self.sigmas = np.random.uniform(-np.sqrt(np.abs(c[1,0])),np.sqrt(np.abs(c[1,0])), size=(2,2)) + c

            self.initials = np.random.rand(clust_num)
            self.initials /= self.initials.sum()
            self.transitions = np.random.rand(clust_num, clust_num)
            self.transitions /= self.transitions.sum(1).reshape(-1,1)
            self.cluster_num = clust_num
        
    def prob_point(self, cluster, data_point):
        if not self.tied:
            return multivariate_normal(mean=self.mus[cluster], cov=self.sigmas[cluster]).pdf(data_point)
        else:
            return multivariate_normal(mean=self.mus[cluster], cov=self.sigmas).pdf(data_point)
    
    def covar_calc(self, x, mu):
        return np.matmul((x - mu).reshape(2,1), (x - mu).reshape(2,1).T)
    
    def forward(self, data):
        log_like = 0
        alphas = np.zeros((len(data), self.cluster_num))
        for t in range(len(alphas)):
            for state in range(self.cluster_num):
                # Initial states
                if t == 0:
                    alphas[t, state] = self.prob_point(state, data[t]) * self.initials[state]
                else:
                    alphas[t, state] = self.prob_point(state, data[t]) * np.sum([alphas[t - 1, j] * self.transitions[j, state] for j in range(self.cluster_num)])
            log_like += np.log(np.sum(alphas[t, :]))
            alphas[t, :] /= np.sum(alphas[t, :])
        return (alphas, log_like/len(data))
    
    def backward(self, data):
        betas = np.zeros((len(data), self.cluster_num))
        betas[-1,:] = 1
        for t in range(len(data) - 2, -1, -1):
            for state in range(self.cluster_num):
                betas[t, state] = np.sum([self.prob_point(j, data[t + 1]) * self.transitions[state, j] * betas[t+1, j] for j in range(self.cluster_num)])
            betas[t, :] /= np.sum(betas[t, :])
        return betas
    
    def train(self, data, iterations):
        for _ in range(iterations):
            xi_mat = np.zeros((len(data) - 1, self.cluster_num, self.cluster_num))
            gamma_mat = np.zeros((len(data), self.cluster_num))
            alphas, _ = self.forward(data)
            betas = self.backward(data)
            ### E Step
            for t in range(len(data)):
                for i in range(self.cluster_num):
                    for j in range(self.cluster_num):
                        # Not the last iteration
                        if t < len(data)-1:
                            xi_mat_num = alphas[t, i] * betas[t+1, j] * self.transitions[i,j] * self.prob_point(j, data[t+1])
                            xi_mat[t,i,j] = xi_mat_num

                            
                    gamma_mat[t, i] = alphas[t, i] * betas[t, i] / np.sum(alphas[t] * betas[t])
                if t < len(data) - 1:
                    xi_mat[t,:] /= xi_mat[t,:].sum()
                    
            ### M Step
            # Update mu, and sigma
            self.sigmas = np.zeros((self.cluster_num, 2, 2))
            for k in range(self.cluster_num):
                self.mus[k, :] = (gamma_mat[:, k].reshape(-1, 1) * data).sum(0) / gamma_mat[:, k].sum()
                self.sigmas[k] = np.sum([gamma_mat[j, k] * self.covar_calc(data[j, :], self.mus[k, :]) for j in range(len(data))], 0)
                if not self.tied:
                    self.sigmas[k] /= np.sum(gamma_mat[:, k])
                # Update initials
                self.initials[k] = gamma_mat[0,k] / gamma_mat[0, :].sum()
            if self.tied:
                self.sigmas = self.sigmas.sum(0) / gamma_mat.sum()
            
            # Update Transitions
            for i in range(self.cluster_num):
                for j in range(self.cluster_num):
                    self.transitions[i,j] = xi_mat[:,i,j].sum() / xi_mat[:, i, :].sum()
            
    def print_params(self, data):
        print('Mus: \n{}\n'.format(self.mus))
        print('Sigmas: \n{}\n'.format(self.sigmas))
        print('Initials: \n{}\n'.format(self.initials))
        print('Transitions: \n{}\n'.format(self.transitions))
        _,log_like = self.forward(data)
        print('Log Like: {}'.format(log_like))

# Originasl gaussian mixed model
class gmm():
    def __init__(self, clust_num, data=None, tied=False, mus=None, sigmas=None, lams=None):
        if mus is not None:
            self.mus = mus
            self.sigmas = sigmas
            self.lam = lams
        else:
            self.tied = False
            self.mus = data[np.random.randint(0,len(data), clust_num)]
            c = np.cov(np.vstack((data[:,0], data[:,1])))
            self.sigmas = np.array([np.random.uniform(-np.sqrt(np.abs(c[1,0])),np.sqrt(np.abs(c[1,0])), size=(2,2)) + c for i in range(clust_num)])
            self.lam = np.random.rand(clust_num)
            self.lam /= self.lam.sum()
            self.cluster_num = clust_num

    def norm(self, x, co, mu):
        return multivariate_normal(mean=mu, cov=co).pdf(x)
    
    def covar_calc(self, x, mu):
        return np.matmul((x - mu).reshape(2,1), (x - mu).reshape(2,1).T)
    
    def train(self, x, iterations):
        for i in range(iterations):
            # E Step
            gamma = np.zeros((self.lam.shape[0], x.shape[0]))
            if not self.tied:
                for n in range(x.shape[0]):
                    for k in range(self.lam.shape[0]):
                        gamma[k,n] = self.lam[k] * self.norm(x[n], self.sigmas[k], self.mus[k])
                    if np.sum(gamma[:,n]) != 0:
                        gamma[:,n] = gamma[:,n] / (np.sum(gamma[:,n]))
                    else:
                        gamma[:,n] = gamma[:,n] / (np.sum(gamma[:,n]) + np.finfo(np.float32).eps)
            else:
                for n in range(x.shape[0]):
                    for k in range(self.lam.shape[0]):
                        gamma[k,n] = self.lam[k] * self.norm(x[n], self.sigmas, self.mus[k])

                    if np.sum(gamma[:,n]) != 0:
                        gamma[:,n] = gamma[:,n] / (np.sum(gamma[:,n]))
                    else:
                        gamma[:,n] = gamma[:,n] / (np.sum(gamma[:,n]) + np.finfo(np.float32).eps)            
            # M Step
            self.sigmas = np.zeros((self.lam.shape[0], 2, 2))
            for k in range(self.lam.shape[0]):
                self.lam[k] = np.sum(gamma[k, :]) / (np.sum(gamma))
                self.mus[k, :] = np.sum(gamma[k, :].reshape(x.shape[0], 1) * x, 0) / (np.sum(gamma[k, :]))
                self.sigmas[k, :] = np.sum([gamma[k, j] * self.covar_calc(x[j, :], self.mus[k, :]) for j in range(x.shape[0])], 0)
                if not self.tied:
                    self.sigmas[k, :] = self.sigmas[k, :] / (np.sum(gamma[k, :]))
            if self.tied:
                self.sigmas = np.sum(self.sigmas, 0) / (np.sum(gamma))
            
    def log_like(self, x):
        ll = 0
        for n in range(x.shape[0]):
            p_x = 0
            for k in range(self.lam.shape[0]):
                if not self.tied:
                    p_x += self.lam[k] * self.norm(x[n], self.sigmas[k], self.mus[k])
                else:
                    p_x += self.lam[k] * self.norm(x[n], self.sigmas, self.mus[k])
            # Deal with the very few initlization cases where p_x = 0
            if p_x == 0:
                ll += np.log(p_x + np.finfo(np.float32).eps)  
            else:
                ll += np.log(p_x)
        return ll / x.shape[0]
