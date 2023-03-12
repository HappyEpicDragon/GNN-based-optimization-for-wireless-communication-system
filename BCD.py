#   BCD aims to solve the channel optimization problem.
#   Class name : BCD
#   parameters:
#       h_dk: channel from BS to user k
#       G: channel from BS to IRS
#       h_rk: channel from IRS to user k
#       Theta: phase-shift matrix of IRS
#       W: transmit beamforming matrix
#
#   member functions:
#       f_1: function which is to be optimized
#           f_1(W, Theta, alpha)
#               W: transmit beamforming
#               Theta: RC matrix
#               alpha: auxiliary variable


import numpy as np


class BCD:
    def __init__(self, h_d, G, h_r, sigma2, P_T, theta, info):
        self.h_d = h_d
        self.h_r = h_r
        self.G = G
        self.sigma2 = sigma2
        self.P_T = P_T
        self.Theta = np.diag(theta)
        self.M = info.IRS_antennas
        self.N = info.BS_antennas
        self.K = info.user_num
        self.alpha, self.beta, self.epsilon, self.W = self.BCD_init()

    @staticmethod
    def BCD_init():
        alpha = None
        beta = None
        epsilon = None
        W = None
        return alpha, beta, epsilon, W

    def optimize(self, eps=1e-10):
        while True:
            last = self.f1()
            self.update_alpha()
            self.update_beta()
            self.update_W()
            self.update_epsilon()
            self.update_theta()
            now = self.f1()
            if np.linalg.norm(now - last) < eps:
                return

    def update_alpha(self):
        gamma = self.gamma()
        K = self.K
        for k in range(K):
            self.alpha[k] = gamma[k]

    def update_beta(self):
        alpha_hat = 1 + self.alpha
        K = self.K
        for k in range(K):
            h_dk = self.h_d[k]
            h_rk = self.h_r[k]
            G_H = self.Hermit(self.G)
            Theta = self.Theta
            h_k = h_dk + np.matmul(np.matmul(G_H, Theta), h_rk)
            h_kH = self.Hermit(h_k)
            alpha_hat_k = alpha_hat[k]
            w_k = self.W[k]
            numerator = np.sqrt(alpha_hat_k) * np.matmul(h_kH, w_k)
            denominator = 0
            for i in range(K):
                w_i = self.W[i]
                t = np.matmul(h_kH, w_i)
                t = np.linalg.norm(t)
                t = np.power(t, 2)
                denominator += t
            denominator += self.sigma2
            beta_k = numerator / denominator
            self.beta[k] = beta_k

    def update_W(self):
        pass

    def update_epsilon(self):
        pass

    def update_theta(self):
        pass

    def f1(self):
        gamma = self.gamma()
        K = self.h_d.shape[0]
        t1 = 0
        for k in range(K):
            alpha_k = self.alpha[k]
            t1 += np.log2(1 + alpha_k)
        t2 = 0
        for k in range(K):
            alpha_k = self.alpha[k]
            t2 -= alpha_k
        t3 = 0
        for k in range(K):
            gamma_k = gamma[k]
            alpha_k = self.alpha[k]
            t3 += (1 + alpha_k) * gamma_k / (1 + gamma_k)
        return t1 + t2 + t3

    def gamma(self):
        gamma = []
        K = self.h_d.shape[0]
        for k in range(K):
            h_dk = self.h_d[k]
            h_rk = self.h_r[k]
            h_dkH = self.Hermit(h_dk)
            h_rkH = self.Hermit(h_rk)
            Theta_H = self.Hermit(self.Theta)
            w_k = self.W[k]
            numerator = np.matmul(h_rkH, Theta_H)
            numerator = np.matmul(numerator, self.G)
            numerator = numerator + h_dkH
            numerator = np.matmul(numerator, w_k)
            numerator = np.linalg.norm(numerator)
            numerator = np.power(numerator, 2)
            denominator = 0
            for i in range(K):
                w_i = self.W[i]
                t = np.matmul(h_rkH, Theta_H)
                t = np.matmul(t, self.G)
                t = t + h_dkH
                t = np.matmul(t, w_i)
                t = np.linalg.norm(t)
                t = np.power(t, 2)
                denominator += t
            denominator += self.sigma2
            gamma_k = numerator / denominator
            gamma.append(gamma_k)
        return np.array(gamma)

    @staticmethod
    def Hermit(matrix):
        return matrix.conjugate().T
