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
        self.alpha, self.beta, self.epsilon, self.W, self.a, self.b = self.BCD_init()

    def BCD_init(self):
        alpha = np.ones([self.K, 1])
        beta = np.ones([self.K, 1])
        epsilon = np.ones([self.K, 1])
        W = np.zeros([self.K, self.M])
        a = None
        b = None
        return alpha, beta, epsilon, W, a, b

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
        Lambda = 1e6
        self.W_o(Lambda)
        p = np.linalg.norm(self.W)
        while p <= self.P_T:
            Lambda /= 2
            self.W_o(Lambda)
            p = np.linalg.norm(self.W)

    def W_o(self, Lambda):
        alpha_hat = 1 + self.alpha
        K = self.K
        for k in range(K):
            h_dk = self.h_d[k]
            h_rk = self.h_r[k]
            G_H = self.Hermit(self.G)
            Theta = self.Theta
            h_k = h_dk + np.matmul(np.matmul(G_H, Theta), h_rk)
            alpha_hat_k = alpha_hat[k]
            beta_k = self.beta[k]
            inv = np.zeros([self.M, self.M])
            for i in range(K):
                h_di = self.h_d[i]
                h_ri = self.h_r[i]
                G_H = self.Hermit(self.G)
                Theta = self.Theta
                h_i = h_di + np.matmul(np.matmul(G_H, Theta), h_ri)
                h_iH = self.Hermit(h_i)
                beta_i = self.beta[i]
                t = np.matmul(h_i, h_iH)
                inv = inv + np.power(beta_i, 2) * t
            inv = Lambda * np.eye(self.M) + inv
            inv = np.linalg.inv(inv)
            w_k = np.sqrt(alpha_hat_k) * beta_k * inv * h_k
            self.W[k] = w_k

    def update_epsilon(self):
        self.a_b()
        alpha_hat = self.alpha + 1
        theta = np.diag(self.Theta)
        theta_H = self.Hermit(theta)
        K = self.K
        for k in range(K):
            alpha_hat_k = alpha_hat[k]
            b_kk = self.b[k, k]
            a_kk = self.a[k, k]
            numerator = np.sqrt(alpha_hat_k) * (b_kk + np.matmul(theta_H, a_kk))
            denominator = 0
            for i in range(K):
                b_ik = self.b[i, k]
                a_ik = self.a[i, k]
                t = b_ik + np.matmul(theta_H, a_ik)
                t = np.linalg.norm(t)
                t = np.power(t, 2)
                denominator += t
            denominator += self.sigma2
            self.epsilon[k] = numerator / denominator

    def update_theta(self):
        K = self.K
        U = np.zeros([self.N, self.N])
        for k in range(K):
            t = self.epsilon[k]
            t = np.power(t, 2)
            t1 = np.zeros([self.N, self.N])
            for i in range(K):
                a_ik = self.a[i, k]
                a_ikH = self.Hermit(a_ik)
                t1 = t1 + np.matmul(a_ik, a_ikH)
            U = U + t * t1
        v = np.zeros([1, self.N])
        alpha_hat = self.alpha + 1
        for k in range(K):
            a_kk = self.a[k, k]
            alpha_hat_k = alpha_hat[k]
            epsilon_k = self.epsilon[k]
            t1 = np.sqrt(alpha_hat_k) * epsilon_k.conjugate() * a_kk
            t2 = np.zeros([1, self.N])
            for i in range(K):
                b_ik = self.b[i, k]
                a_ik = self.a[i, k]
                t2 = t2 + b_ik.conjugate() * a_ik
            t2 = t2 * np.power(epsilon_k, 2)
            v = v + t1 - t2
        N = self.N
        inv = np.eye(N) + U
        inv = np.linalg.inv(inv)
        theta = np.matmul(inv, v)
        self.Theta = np.diag(theta)

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

    def a_b(self):
        a = []
        b = []
        K = self.K
        for i in range(K):
            w_i = self.W[i]
            a_i = []
            b_i = []
            for k in range(K):
                h_rk = self.h_r[k]
                h_rkH = self.Hermit(h_rk)
                a_ik = np.matmul(np.diag(h_rkH), self.G)
                a_ik = np.matmul(a_ik, w_i)
                h_dk = self.h_d[k]
                h_dkH = self.Hermit(h_dk)
                b_ik = np.matmul(h_dkH, w_i)
                a_i.append(a_ik)
                b_i.append(b_ik)
            a.append(a_i)
            b.append(b_i)
        self.a = np.array(a)
        self.b = np.array(b)

    @staticmethod
    def Hermit(matrix):
        return matrix.conjugate().T
