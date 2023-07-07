import copy
import time

import numpy as np
import scipy
from numpy import ndarray

from Particle import Particle, mean_prior, theta_prior, evaluation_likelihood
from Util import sequence_of_exponents


class Posterior(object):

    exponent_like: ndarray
    ess: ndarray
    norm_cost: ndarray
    particle: ndarray
    all_particles: ndarray
    all_weights_unnorm: ndarray
    all_weights: ndarray

    n_iter: int

    grid_theta: ndarray
    theta_posterior: ndarray
    map_theta: float
    pm_theta: float
    ml_theta: float

    best_iter: int
    map_mean_eb: float
    pm_mean_eb: float
    ess_eb: float

    map_mean: float
    pm_mean: float
    mean_posterior: ndarray
    mean_eb_posterior: ndarray

    cpu_time: float

    ess_max: float
    ess_min: float
    delta_max: float
    delta_min: float
    n_point_interp: int

    def __init__(self, cfg=None):

        cfg = cfg or {}

        self.n_particles = int(cfg.get('n_particles', 10))
        self.theta_eff = cfg.get('theta_eff')
        self.sourcespace = cfg.get('sourcespace')
        self.data = cfg.get('data')
        self.sequence_evolution = cfg.get('sequence_evolution')
        self.method = cfg.get('method')
        self.verbose = cfg.get('verbose')

        self.exponent_like = np.array([0.0, 0.0])
        self.ess = np.full(self.n_particles, self.n_particles)
        self.norm_cost = np.array([1])
        self.particle = np.array([Particle(cfg=cfg) for _ in range(self.n_particles)])
        self.all_particles = np.array([self.particle])
        self.all_weights_unnorm = np.ones((1, self.n_particles))
        self.all_weights = 1 / self.n_particles * np.ones((1, self.n_particles))
        self.n_iter = None
        self.grid_theta = None
        self.theta_posterior = None
        self.map_theta = None
        self.pm_theta = None
        self.ml_theta = None
        self.best_iter = None
        self.map_mean_eb = None
        self.pm_mean_eb = None
        self.ess_eb = None
        self.map_mean = None
        self.pm_mean = None
        self.mean_posterior = None
        self.mean_eb_posterior = None
        self.cpu_time = None
        self.ess_max = 0.89
        self.ess_min = 0.8
        self.delta_max = 1e-1
        self.delta_min = 1e-4
        self.n_point_interp = 0
        self.ancestors = np.c_[np.arange(self.n_particles), np.arange(self.n_particles)]

        if self.method == 'PM':
            self.best_iter = None
            self.map_mean_eb = None
            self.pm_mean_eb = None
            self.ess_eb = None

        self.map_mean = None
        self.pm_mean = None
        self.mean_posterior = None
        self.mean_eb_posterior = None

        self.cpu_time = None

        self.ess_max = 0.89
        self.ess_min = 0.8
        self.delta_max = 1e-1
        self.delta_min = 1e-4
        self.n_point_interp = 0
        self.ancestors = np.c_[np.arange(self.n_particles), np.arange(self.n_particles)]

    def metropolis_hastings(self):
        exponent_like_last = self.exponent_like[-1]
        self.particle = [p.mh_mean(self.sourcespace, self.data, exponent_like_last) for p in self.particle]

        if self.method == 'FB':
            self.particle = [p.mh_theta(self.sourcespace, self.data, exponent_like_last) for p in self.particle]

        return self


    def importance_sampling(self, next_alpha):
        weight_u = np.zeros(self.n_particles)
        for idx, _p in enumerate(self.particle):
            new_like = evaluation_likelihood(_p.mean, _p.theta, self.sourcespace, self.data, next_alpha)
            weight_upgrade = 0 if _p.like == 0 else new_like / _p.like
            weight_u[idx] = _p.weight_u * weight_upgrade
            _p.like = new_like
        weight = np.divide(weight_u, np.sum(weight_u))

        for idx, (_p, w_u, w) in enumerate(zip(self.particle, weight_u, weight)):
            _p.weight_u = w_u
            _p.weight = w

        self.ess = np.append(self.ess, 1 / np.sum(np.power(weight, 2)))
        self.norm_cost = np.append(self.norm_cost, 1 / self.n_particles * np.sum(weight_u))

        return self

    def resampling(self):
        if self.ess[-1] < 0.5 * self.n_particles:
            new_ancestors = np.zeros(self.n_particles)
            self.ess[-1] = self.n_particles
            auxiliary_particle = copy.deepcopy(self.particle)
            u = np.random.rand()
            for idx, _p in enumerate(self.particle):
                threshold = (u + idx) / self.n_particles
                sum_weight = 0
                j = -1
                while sum_weight < threshold and j < self.n_particles - 1:
                    j += 1
                    sum_weight += self.particle[j].weight
                self.particle[idx] = copy.deepcopy(auxiliary_particle[j])
                new_ancestors[idx] = self.ancestors[j, -1]
            for _p in self.particle:
                _p.weight = 1 / self.n_particles
                _p.weight_u = self.norm_cost[-1] / self.n_particles
        else:
            new_ancestors = self.ancestors[:, -1]
        self.ancestors = np.c_[self.ancestors, new_ancestors]
        return self

    def evolution_exponent(self):
        if self.sequence_evolution is None:
            if self.exponent_like[-1] == 1:
                next_exponent = 1.1
            else:
                delta_a = self.delta_min
                delta_b = self.delta_max
                is_last_operation_increment = False
                delta = self.delta_max
                next_exponent = self.exponent_like[-1] + delta
                self_aux = copy.deepcopy(self)
                self_aux.ess[-1] = 0
                iterations = 1
                while not self.ess_min <= self_aux.ess[-1] / self.ess[-1] <= self.ess_max and iterations < 1e2:
                    self_aux = copy.deepcopy(self)
                    self_aux = self_aux.importance_sampling(next_exponent)

                    if self_aux.ess[-1] / self.ess[-1] > self.ess_max:
                        delta_a = delta
                        delta = min((delta_a + delta_b) / 2, self.delta_max)
                        is_last_operation_increment = True
                        if self.delta_max - delta < self.delta_max / 100:
                            next_exponent = self.exponent_like[-1] + delta
                            self_aux = self_aux.importance_sampling(next_exponent)
                            if next_exponent >= 1:
                                next_exponent = 1
                                self_aux.ess[-1] = self.ess[-1] * (self.ess_max + self.ess_min) / 2
                            break
                    else:
                        if self_aux.ess[-1] / self.ess[-1] < self.ess_min:
                            delta_b = delta
                            delta = max((delta_a + delta_b) / 2, self.delta_min)
                            if delta - self.delta_min < self.delta_min / 10 or \
                                    (iterations > 1 and is_last_operation_increment):
                                next_exponent = self.exponent_like[-1] + delta
                                self_aux = self_aux.importance_sampling(next_exponent)
                                if next_exponent >= 1:
                                    next_exponent = 1
                                    self_aux.ess[-1] = self.ess[-1] * (self.ess_max + self.ess_min) / 2
                                break
                                is_last_operation_increment = False
                    next_exponent = self.exponent_like[-1] + delta
                    if next_exponent >= 1:
                        next_exponent = 1
                        self_aux.ess[-1] = self.ess[-1] * (self.ess_max + self.ess_min) / 2
                    iterations += 1
        else:
            next_exponent = sequence_of_exponents(self.sequence_evolution, 1)[len(self.exponent_like)]

        return next_exponent

    def perform_smc(self):
        start_time = time.time()

        if self.method == 'EM':
            self.map_theta_eval()

        n = 0
        if self.verbose:
            print(f'iter:{n} -- exp: {self.exponent_like[n]}')
            
        self = self.importance_sampling(self.exponent_like[-1])
        self.all_particles = np.concatenate([self.all_particles, np.array([self.particle])], axis=0)
        self.all_weights_unnorm = np.concatenate([self.all_weights_unnorm,
                                                  np.array([[_p.weight_u for _p in self.particle]])],
                                                 axis=0)
        self.all_weights = np.concatenate([self.all_weights, np.array([[_p.weight for _p in self.particle]])])

        n = 1
        while self.exponent_like[-1] <= 1:
            self = self.metropolis_hastings()
            self.exponent_like = np.append(self.exponent_like, self.evolution_exponent())
            self = self.importance_sampling(self.exponent_like[-1])
            self = self.resampling()
            self.vector_post()
            self.store_iteration()
            self.mean_estimates()
            if self.verbose:
                print(f'iter:{n} -- exp: {"{:.4f}".format(self.exponent_like[n])} \n'
                      f'MAP mean: {self.map_mean} -- PM mean: {self.pm_mean}')
            n += 1
        self.n_iter = n

        if self.method == 'PM':
            self = self.compute_big_posterior()
            self.pm_eb_eval()
            self.vector_post()
            self.mean_estimates()
            if self.verbose:
                print(f'MAP mean: {self.map_mean} -- PM mean: {self.pm_mean}')

        self.theta_estimates()
        if self.verbose:
            print(f'MAP theta: {self.map_theta} -- PM theta: {self.pm_theta}')

        self.cpu_time = time.time() - start_time
        if self.verbose:
            print('\n-- time for execution: %s (s) --' % self.cpu_time)

        return self

    def map_theta_eval(self):
        self.grid_theta = np.linspace(0.5 * self.theta_eff, 10 * self.theta_eff, 1000)
        theta_prior = scipy.stats.gamma.pdf(self.grid_theta, a=2, scale=4 * self.grid_theta[0])
        self.theta_posterior = self.integral(theta=self.grid_theta) * theta_prior
        self.theta_posterior /= np.trapz(self.theta_posterior, self.grid_theta)
        self.theta_eff = self.grid_theta[np.argmax(self.theta_posterior)]

    def integral(self, theta):
        res = 0
        for _m in self.sourcespace:
            res += evaluation_likelihood(_m, theta, self.sourcespace, self.data, exponent_like=1) * mean_prior(_m)
        return 1 / len(self.sourcespace) * res

    def compute_big_posterior(self):
        particle_aux = []
        integral_weight_u = []
        norm_cost = self.norm_cost[2: self.n_iter]
        exponent_like = self.exponent_like[2: self.n_iter]

        all_theta = self.theta_eff / np.sqrt(exponent_like)

        delta_std = np.zeros(len(all_theta))
        delta_std[0] = abs(all_theta[0] - all_theta[1])
        delta_std[-1] = abs(all_theta[-2] - all_theta[-1])
        for i in range(2, len(all_theta)):
            delta_std[i - 1] = abs(all_theta[i - 2] - all_theta[i])
        k = np.power(np.power(2 * np.pi * np.square(self.theta_eff), exponent_like - 1) * exponent_like,
                     self.data.shape[0] / 2)
        weight_upgrade = 0.5 * delta_std * k * theta_prior(all_theta, self.theta_eff) * norm_cost

        for t_idx in range(self.n_iter - 2):
            for p_idx in range(self.n_particles):
                integral_weight_u = np.append(integral_weight_u,
                                              self.all_weights[t_idx + 2, p_idx] * weight_upgrade[t_idx])
                particle_aux = np.append(particle_aux, self.all_particles[t_idx + 2, p_idx])

        integral_weight = integral_weight_u / np.sum(integral_weight_u)

        self.particle = copy.deepcopy(particle_aux)
        for idx, _p in enumerate(self.particle):
            _p.weight_u = integral_weight_u[idx]
            _p.weight = integral_weight[idx]

        self.ml_theta = all_theta[np.argmax(norm_cost * k)]
        self.theta_posterior = theta_prior(all_theta, self.theta_eff) * norm_cost * k
        
        self.best_iter = np.argmax(self.theta_posterior)
        self.grid_theta = np.unique(
            np.sort(np.append(all_theta, np.linspace(np.min(all_theta), np.max(all_theta), int(self.n_point_interp)))))
        self.theta_posterior = scipy.interpolate.interp1d(all_theta, self.theta_posterior, kind='linear')(
            self.grid_theta)
        integral = 0.5 * np.sum(
            (self.theta_posterior[:-1] + self.theta_posterior[1:]) * np.abs(self.grid_theta[:-1] - self.grid_theta[1:]))
        self.theta_posterior /= integral

        self.ess_eb = self.ess[self.best_iter]
        self.ess = 1 / np.sum(integral_weight ** 2)

        return self


    def mean_estimates(self):
        x_mean = np.linspace(-5, 5, 10000)
        self.mean_posterior = scipy.stats.gaussian_kde(self.vector_mean, weights=self.vector_weight).pdf(x_mean)
        self.mean_posterior /= np.sum(self.mean_posterior)

        self.pm_mean = np.sum([_p.mean * _p.weight for _p in self.particle])
        self.map_mean = x_mean[np.argmax(self.mean_posterior)]

    
    def pm_eb_eval(self):
        x_mean = np.linspace(-5, 5, 10000)
        vector_mean = [p.mean for p in self.all_particles[self.best_iter]]
        vector_weight = [p.weight for p in self.all_particles[self.best_iter]]

        self.mean_eb_posterior = scipy.stats.gaussian_kde(vector_mean, weights=vector_weight).pdf(x_mean)
        self.mean_eb_posterior /= np.sum(self.mean_eb_posterior)

        self.pm_mean_eb = np.dot(vector_mean, vector_weight)

        self.map_mean_eb = x_mean[np.argmax(self.mean_eb_posterior)]


    def theta_estimates(self):
        self.pm_theta = 0
        if self.method in ['PM', 'EM']:
            # posterior mean
            delta = np.diff(self.grid_theta)
            self.pm_theta = 0.5 * np.sum((self.grid_theta[:-1] + self.grid_theta[1:]) * self.theta_posterior[:-1] * delta)

        if self.method == 'PM':
            # maximum a posteriori
            self.map_theta = self.grid_theta[np.argmax(self.theta_posterior)]

        if self.method == 'EM':
            # maximum a posteriori
            self.map_theta = self.theta_eff

        if self.method == 'FB':
            self.grid_theta = np.linspace(0, 1, 10000)
            vector_theta = np.array(self.vector_theta)
            vector_weight = np.array(self.vector_weight)
            theta_posterior = scipy.stats.gaussian_kde(vector_theta, weights=vector_weight).pdf(self.grid_theta)

            # posterior mean
            self.pm_theta = np.dot(vector_weight, vector_theta)

            # maximum a posteriori
            self.map_theta = self.grid_theta[np.argmax(theta_posterior)]


    def vector_post(self):
        num_particles = len(self.particle)
        self.vector_mean = [0] * num_particles
        self.vector_theta = [0] * num_particles
        self.vector_weight = [0] * num_particles
        self.vector_weight_u = [0] * num_particles

        for idx, _p in enumerate(self.particle):
            self.vector_mean[idx] = _p.mean
            self.vector_theta[idx] = _p.theta
            self.vector_weight[idx] = _p.weight
            self.vector_weight_u[idx] = _p.weight_u


    def store_iteration(self):
        self.all_particles = np.vstack([self.all_particles, self.particle])
        self.all_weights_unnorm = np.vstack([self.all_weights_unnorm, self.vector_weight_u])
        self.all_weights = np.vstack([self.all_weights, self.vector_weight])
