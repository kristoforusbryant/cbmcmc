"""
MCMC for estimating Gaussian Graphical Model Posterior restricted to the Cycle Space 

The Python code below are functions used to estimate the posterior distribution of 
Gaussian Graphical Models using the Markov Chain Monte Carlo (MCMC) algorithm.
In particular, the code contains the Cycle Basis Proposal used to estimate Gaussian Graphical Models 
that are restricted to the Cycle Space.
"""

import time, tqdm, pickle

import numpy as np
from scipy.special import loggamma
from itertools import combinations
from sklearn.metrics import roc_auc_score, f1_score

from utils.myGraph import Graph
from utils.laplace_approximation import laplace_approx
from utils.diagnostics import IAT, str_list_to_adjm

# Prior P(G)
class Uniform:
    """
    Uniform prior over all graphs having n nodes i.e. the probability of each graph is 1 / (2 ** (n * (n - 1) / 2).
    """
    def __init__(self, n, Param):
        self._n = n
        self._Param = Param

    __name__ = 'uniform'

    def Sample(self):
        param = self._Param(self._n)
        triu = np.triu_indices(self._n,1)
        for i, j in list(zip(triu[0], triu[1])):
            if np.random.uniform() > .5:
                    param.AddEdge(i,j)
        return param

    def PDF(self, param):
        return 0

    def ParamType(self):
        return self._Param.__name__

class EdgeInclusion:
    """
    Prior over the set of all graphs having n nodes tha is induced by a Bernoulli prior with parameter p 
    on the inclusion of every edge of the graph.
    """
    def __init__(self, n, Param, p=0.5):
        self._n = n
        self._m = (n) * (n - 1) // 2
        self._Param = Param
        self._p = p

    __name__ = 'edge-inclusion'

    def Sample(self):
        param = self._Param(self._n)
        triu = np.triu_indices(self._n,1)
        for i, j in list(zip(triu[0], triu[1])):
            if np.random.uniform() > self._p:
                 param.AddEdge(i,j)
        return param

    def PDF(self, param):
        k = param._size
        return k * np.log(self._p) + (self._m - k) * np.log(1 - self._p)

    def ParamType(self):
        return self._Param.__name__

# Proposal q(G -> G')
class StarCycleBases:
    """
    Proposal that moves one-cycle-basis-at-a-time given that the prior on the spanning trees 
    is uniform over all star trees.
    """
    def __init__(self, n, Param):
        # rv_size: distribution of the basis size
        self._n = n
        self._Param = Param
        self._last_change = None

    __name__ = 'star-cycle-bases'

    def Sample(self, param):
        nodes = np.random.choice(np.arange(self._n), 3, replace=False)
        self._last_change = nodes
        for i, j in combinations(nodes, 2):
            param.FlipEdge(i, j)
        return param

    def Toggle(self, param):
        for i, j in combinations(self._last_change, 2):
            param.FlipEdge(i, j)
        return param

    def PDF_ratio(self, p_):
        return 0

    def PDF(self, p, p_):
        return 0

    def ParamType(self):
        return self._Param.__name__

class OneEdge:
    """
    Standard graphical model proposal of moving one-edge-at-a-time.
    """
    def __init__(self, n, Param):
        self._n = n
        self._Param = Param
        self._last_change = None

    __name__ = 'one-edge'

    def Sample(self, param):
        nodes = np.random.choice(np.arange(self._n), 2, replace=False)
        self._last_change = nodes
        for i, j in combinations(nodes, 2):
            param.FlipEdge(i, j)
        return param

    def Toggle(self, param):
        for i, j in combinations(self._last_change, 2):
            param.FlipEdge(i, j)
        return param

    def PDF_ratio(self, p_):
        return 0

    def PDF(self, p, p_):
        return 0

    def ParamType(self):
        return self._Param.__name__

# Marginal Likelihood P(X | G)
class GW_Ratio:
    """
    Computation of the marginal likelihood ratio of two graphs. 
    The marginal likelihood is computed in two parts:
    - The ratio of normalising constant of the G-Wishart prior is computed using the approximation in (Mohammadi et al., 2021, arXiv:1706.04416), whilst
    - the normalising constants of the G-Wishart posteriors is computed using the Laplace Approximation in (Lenkoski & Dobra, 2011, doi.org/10.1198/jcgs.2010.08181).
    """
    def __init__(self, data, delta, D, Param, alpha=1):
        self._D = D
        self._D_star = D + data.transpose() @ data
        self._delta = delta
        self._delta_star = delta +  data.shape[0]
        self._Param = Param
        self._alpha = alpha
        self._lookup = {} # graph ID: log_IG_post

    def log_IG_prior_ratio(self, param_, diff):
        # param_ is the proposed
        res = 0
        for i, j in combinations(diff, 2):
            d = 0 # 2-edges path between i and j
            for k in param_._dol[i]:
                if j in param_._dol[k]:
                    d += 1
            sign = -1 if (j in param_._dol[i]) else 1
            gamma_ratio = loggamma((self._delta + d) / 2) - loggamma((self._delta + d + 1) / 2)
            res += sign * (-np.log(2 * np.sqrt(np.pi)) + gamma_ratio)
            param_.FlipEdge(i, j)
        return res

    def log_IG_post(self, param):
        return laplace_approx(param._dol, self._delta_star, self._D_star)

    def ParamType(self):
        return self._Param.__name__

class MCMC_Sampler_Ratio:
    def __init__(self, prior, proposal, likelihood, data, outfile=""):
        self.prior = prior
        self.prop = proposal
        self.lik = likelihood
        self.data = data
        self.res = {'SAMPLES':[],
                    'ALPHAS':[],
                    'PARAMS':[],
                    'PARAMS_PROPS': [],
                    'ACCEPT_INDEX':[],
                    'LIK_R':[],
                    'PRIOR':[],
                    'PRIOR_':[],
                    'U':[]
                     }

        self.lookup = {} # dict of dicts
        self.lookup_count = 0
        self.time = 0.0
        self.iter = 0
        self.outfile = outfile

    def run(self, it=10000, fixed_init=None):
        tic = time.time()

        # Initialisation
        if fixed_init is not None:
            params = fixed_init
        else:
            params = self.prior.Sample()

        id_p = params.GetID()
        self.lookup[id_p] = self.lik.log_IG_post(params)
        prior_p = self.prior.PDF(params)

        # Iterate
        for i in tqdm.tqdm(range(it)):
            params = self.prop.Sample(params)

            id_p_ = params.GetID()
            self.res['PARAMS'].append(id_p_)
            prior_p_ = self.prior.PDF(params)
            if id_p_ not in self.lookup:
                self.lookup[id_p_] = self.lik.log_IG_post(params)

            lik_r = self.lookup[id_p_] - self.lookup[id_p] - self.lik.log_IG_prior_ratio(params, self.prop._last_change)
            self.prop.Toggle(params) # log_IG_prior_ratio Toggles param back from proposed to current, this line Toggles it back
            prior_r = prior_p_ - prior_p
            prop_r = self.prop.PDF_ratio(params)
            alpha = lik_r + prior_r + prop_r

            self.res['PARAMS_PROPS'].append({'PRIOR': prior_p_})
            self.res['LIK_R'].append(lik_r)
            self.res['PRIOR_'].append(prior_p_)
            self.res['ALPHAS'].append(alpha)

            u = np.log(np.random.uniform())
            self.res['U'].append(u)
            if u < alpha:
                self.res['ACCEPT_INDEX'].append(1)
                self.res['SAMPLES'].append(id_p_)
                self.res['PRIOR'].append(prior_p_)
                id_p = id_p_
                prior_p = prior_p_
            else:
                self.res['ACCEPT_INDEX'].append(0)
                self.res['SAMPLES'].append(id_p)
                self.res['PRIOR'].append(prior_p)
                params = self.prop.Toggle(params)

        self.time = self.time + time.time() - tic # in seconds
        self.iter = self.iter + it
        self.last_params = params.copy()
        return 0

    def save_object(self):
        with open(self.outfile, 'wb') as handle:
            pickle.dump(self, handle)
        return 0

    def continue_chain(self, it):
        self.run(it, fixed_init= self.last_params)
        return 0

    def get_summary(self, true_g, b, inc_distances=True, thin=1, acc_scaled_size=None):
        return MCMC_summary(self, true_g, b=b, inc_distances=inc_distances, thin=thin, acc_scaled_size=acc_scaled_size)

class MCMC_summary():
    def __init__(self, sampler, true_g, b=0, alpha=.5, inc_distances=True, thin=100, acc_scaled_size=None):
        self.time = sampler.time
        self.iter = sampler.iter
        self.last_params = sampler.last_params
        self.sizes = list(map(lambda s: np.sum(self._str_to_int_list(s)), sampler.res['SAMPLES'][b::thin]))
        self.summary = self._get_summary(sampler, b, inc_distances=inc_distances, thin=thin, acc_scaled_size=acc_scaled_size)
        self.adjm = str_list_to_adjm(sampler.data.shape[1], sampler.res['SAMPLES'][b::thin])

        self.AUCs = [self._get_AUCs(true_g, self._get_median_graph(sampler, true_g, threshold, b=b, thin=thin)) \
                            for threshold in [.25, .5, .75]]

    def _get_median_graph(self, sampler, true_g, alpha=.5, b=0, thin=1):
        adjm = str_list_to_adjm(len(true_g), sampler.res['SAMPLES'][b::thin])
        return (adjm > alpha).astype(int)

    def _get_accuracies(self, g, md):
        l1= np.array(g.GetBinaryL(), dtype=bool)
        triu = np.triu_indices(len(g), 1)
        l2 = np.array(md[triu], dtype=bool)

        TP = np.logical_and(l1, l2).astype(int).sum()
        TN = np.logical_and(np.logical_not(l1), np.logical_not(l2)).astype(int).sum()
        FP = np.logical_and(np.logical_not(l1), l2).astype(int).sum()
        FN = np.logical_and(l1, np.logical_not(l2)).astype(int).sum()

        assert(TP + TN + FP + FN == len(l1))
        assert(TP + FP == l2.astype(int).sum())
        assert(TN + FN == np.logical_not(l2).astype(int).sum())

        return TP, TN, FP, FN

    def _get_AUCs(self, g, md):
        triu = np.triu_indices(len(g), 1)
        try:
            auc = roc_auc_score(g.GetBinaryL(), np.array(md[triu], dtype=bool))
        except ValueError:
            auc = np.nan
        return auc

    def _get_F1s(self, g, md):
        triu = np.triu_indices(len(g), 1)
        return f1_score(g.GetBinaryL(), np.array(md[triu], dtype=bool))

    def _str_to_int_list(self, s):
        return np.array(list(s), dtype=int)

    def _get_distances(self, str_list, dist, values_to_save=5000):
        """diameter of a unique graph_id list according to specified dist: str * str -> int"""
        a = np.sort([ dist(s1, s2) for s1, s2 in combinations(str_list, 2) ])
        return a

    def _get_generalised_variance(self, str_list):
        """Generalised variance as graph diversity measure (Scutari, 2013, doi:10.1214/13-BA819)"""
        X =  np.array([np.array(list(s), dtype=int) for s in str_list])
        X = X - np.mean(X, axis=0)
        cov = (np.transpose(X) @ X) / (X.shape[0] - 1)
        return np.linalg.det(cov)

    def _get_total_variance(self, str_list):
        """Total variance as graph diversity measure (Scutari, 2013, doi:10.1214/13-BA819)"""
        X = np.array([np.array(list(s), dtype=int) for s in str_list])
        X = X - np.mean(X, axis=0)
        cov = (np.transpose(X) @ X) / (X.shape[0] - 1)
        return np.trace(cov)

    def _get_summary(self, sampler, b=0, inc_distances=True, thin=1, acc_scaled_size=None):
        sizes = list(map(lambda s: np.sum(self._str_to_int_list(s)), sampler.res['SAMPLES']))[b::thin]

        d = {}

        print('Calculating IATs...')
        d['IAT_sizes'] = IAT(sizes)

        d['accept_rate'] = np.sum(sampler.res['ACCEPT_INDEX']) / len(sampler.res['ACCEPT_INDEX'])


        d['states_visited'] = 0
        visited = set()
        for st in sampler.res['SAMPLES'][b::thin]:
            if st not in set():
                d['states_visited'] += 1
                visited.add(st)

        d['states_considered'] = 0
        visited = set()
        for st in sampler.res['PARAMS'][b::thin]:
            if st not in set():
                d['states_considered'] += 1
                visited.add(st)

        print('Calculating variances...')
        if acc_scaled_size:
            print('Calculating acc_scaled distances...')

            accept_idx = np.where(sampler.res['ACCEPT_INDEX'])[0]
            first_x_idx = accept_idx[:acc_scaled_size] + 1 # plus 1 for next proposed
            last_x_idx = accept_idx[-acc_scaled_size:] + 1

             # Edge case where the last iteration is accepted (+1 cause out of index)
            first_x_idx = first_x_idx[first_x_idx < sampler.iter]
            last_x_idx = last_x_idx[last_x_idx < sampler.iter]

            first_x = np.array(sampler.res['PARAMS'])[first_x_idx]
            last_x = np.array(sampler.res['PARAMS'])[last_x_idx]
            d['as_start_tvar'] = self._get_total_variance(first_x)
            d['as_end_tvar'] = self._get_total_variance(last_x)

        d['time'] = sampler.time

        return d
