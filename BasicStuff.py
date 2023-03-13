import numpy as np
import torch
from torch.autograd import Variable


class info_container:
    def __init__(self, user_num, positions,
                 BS_antennas, IRS_antennas,
                 pilot_length,
                 updating_layer_num,
                 uplink_power, downlink_power,
                 uplink_noise_power, downlink_noise_power,
                 requires_location):
        self.user_num = user_num
        self.user_array = []
        for i in range(user_num):
            User = self.Objective(positions[2 + i][0], positions[2 + i][1], positions[2 + i][2])
            self.user_array.append(User)
        self.BS_antennas = BS_antennas
        self.BS = self.Objective(positions[0][0], positions[0][1], positions[0][2], )
        self.IRS_antennas = IRS_antennas
        self.IRS = self.Objective(positions[1][0], positions[1][1], positions[1][2], )
        self.pilot_length = pilot_length
        self.updating_layer_num = updating_layer_num
        self.uplink_power = uplink_power
        self.downlink_power = downlink_power
        self.uplink_noise_power = uplink_noise_power
        self.downlink_noise_power = downlink_noise_power
        self.requires_location = requires_location

    class Objective:
        def __init__(self, xx, yy, zz):
            self.xx = xx
            self.yy = yy
            self.zz = zz


class WirelessSystem:
    def __init__(self, info):
        self.IRS = info.IRS
        self.BS = info.BS
        self.user_array = info.user_array
        self.user_num = info.user_num
        self.M = info.BS_antennas  # BS antennas
        self.N = info.IRS_antennas  # passive reflective elements
        self.pilot_length = info.pilot_length
        self.A = None
        self.h_d = None
        self.h_r = None
        self.G = None
        self.v = []
        for i in range(self.N):
            v_i = np.exp(np.random.rand() * 2 * np.pi - np.pi) ** 1j
            self.v.append(v_i)
        self.v = np.array([self.v])
        self.uplink_power = info.uplink_power
        self.uplink_noise_power = info.uplink_noise_power
        self.requires_location = info.requires_location
        self.transmit_parameter_generator()

    @staticmethod
    def distanceBetween(A, B):
        return np.sqrt(
            (A.xx - B.xx) ** 2 +
            (A.yy - B.yy) ** 2 +
            (A.zz - B.zz) ** 2
        )

    @staticmethod
    def path_loss(distance, arg="r"):
        if arg == "r":
            dB = 30 + 22 * np.log(distance / 1000)  # dB
            coe = np.exp(dB / 10 * np.log(10))
            return coe
        else:
            dB = 32.6 + 36.7 * np.log(distance / 1000)  # dB
            coe = np.exp(dB / 10 * np.log(10))
            return coe

    @staticmethod
    def i_1(n):
        return (n - 1) % 10

    @staticmethod
    def i_2(n):
        n = (n - 1) / 10
        return n - n % 1

    def a_IRS(self, k):
        k = k - 1
        if k >= 0:
            dd = self.distanceBetween(self.user_array[k], self.IRS)
            sin_times_cos = (self.user_array[k].yy - self.IRS.yy) / dd
            sin = (self.user_array[k].zz - self.IRS.zz) / dd
        else:
            dd = self.distanceBetween(self.BS, self.IRS)
            sin_times_cos = (self.BS.yy - self.IRS.yy) / dd
            sin = (self.BS.zz - self.IRS.zz) / dd
        a_IRS = []
        for i in range(1, self.N + 1):
            a_i = np.exp(np.pi * ((self.i_1(i) * sin_times_cos) +
                                  (self.i_2(i) * sin))) ** 1j
            a_IRS.append(a_i)
        return np.array([a_IRS])

    def a_BS(self):
        a_BS = []
        dd = self.distanceBetween(self.IRS, self.BS)
        for i in range(self.M):
            a_i = np.exp(np.pi * i * ((self.IRS.xx - self.BS.xx) / dd))
            a_BS.append(a_i)
        return np.array([a_BS])

    @staticmethod
    def complex_Gaussian(delta=1):
        real = np.random.normal(0, delta / 2)
        imag = np.random.normal(0, delta / 2)
        return real + imag * 1j

    @staticmethod
    def Gram_Schmidt(M):
        def proj(x, u):
            u = unit_vec(u)
            return np.dot(x, u.conj()) * u

        def unit_vec(x):
            return x / np.linalg.norm(x)

        M = np.atleast_2d(M)

        if len(M) == 0:
            return np.array([])

        if len(M) == 1:
            return unit_vec(M)

        shape = np.shape(M)
        output = np.zeros(shape) + 0j
        decrease = 0
        for i in range(shape[0]):
            for j in range(i):
                decrease = decrease + proj(M[i], output[j])
            output[i] = M[i] - decrease
            decrease = 0

        for i in range(shape[0]):
            output[i] = unit_vec(output[i])
        return output

    @staticmethod
    def vectorize_2D(Matrix):
        shape = Matrix.shape
        v = np.zeros([1, shape[0] * shape[1]])
        for i in range(shape[0]):
            v[0, i * shape[1]:(i + 1) * shape[1]] = Matrix[i, :]
        return v

    def transmit_parameter_generator(self):
        RicianFactor = 10  # set number
        h_r_1_LOS = self.a_IRS(1)
        h_r_2_LOS = self.a_IRS(2)
        h_r_3_LOS = self.a_IRS(3)
        h_r_LOS = np.concatenate([np.concatenate([h_r_1_LOS, h_r_2_LOS], axis=0), h_r_3_LOS], axis=0)
        G_LOS = self.a_BS().T * self.a_IRS(-1).conj()
        h_r_NLOS = np.zeros([self.user_num, self.N]) + 0j
        for i in range(self.N):
            for j in range(self.user_num):
                h_r_NLOS[j, i] = self.complex_Gaussian()
        G_NLOS = np.zeros([self.M, self.N]) + 0j
        for i in range(self.M):
            for j in range(self.N):
                G_NLOS[i, j] = self.complex_Gaussian()
        h_r = []
        for i in range(self.user_num):  # 3 is the number of users
            h_r_i = self.path_loss(self.distanceBetween(self.user_array[i], self.IRS)) * (
                    np.sqrt(RicianFactor / (1 + RicianFactor)) * h_r_LOS[i] + np.sqrt(1 / (1 + RicianFactor)) *
                    h_r_NLOS[i])
            h_r.append(h_r_i)
        np.array([h_r])  # finish h_r
        G = self.path_loss(self.distanceBetween(self.IRS, self.BS), "r") * (np.sqrt(RicianFactor / (1 + RicianFactor)) *
                           G_LOS + np.sqrt(1 / (1 + RicianFactor)) * G_NLOS)  # finish G
        h_d = np.zeros([self.user_num, self.M]) + 0j
        for i in range(self.user_num):
            for j in range(self.M):
                h_d[i, j] = self.complex_Gaussian()
        for i in range(self.user_num):
            h_d[i] = h_d[i] * self.path_loss(self.distanceBetween(self.user_array[i], self.BS), "not r")
        # finish h_d
        A = []
        for i in range(self.user_num):
            diag = np.diag(h_r[i])
            A_k = np.matmul(G, diag)
            A.append(A_k)
        A = np.array(A)
        self.G = G
        self.h_d = h_d
        self.h_r = h_r
        self.A = A

    def pilot_signal_generator(self):
        pilot = np.zeros([self.user_num, self.pilot_length]) + 0j
        for i in range(self.user_num):
            for j in range(self.pilot_length):
                pilot[i, j] = np.random.rand() + np.random.rand() * 1j

        pilot = self.Gram_Schmidt(pilot)
        pilot = np.expand_dims(pilot, 1)
        upPower = np.exp(self.uplink_power / 10 * np.log(10)) / 1000
        cof = self.pilot_length * upPower ** (1 / 2)
        pilot = pilot * cof
        Y = np.zeros([self.user_num, self.M, self.pilot_length])
        upNoisePower = np.exp(self.uplink_noise_power / 10 * np.log(10)) / 1000
        for i in range(self.user_num):
            t = np.array([self.h_d[i]]).T + np.matmul(self.A[i], self.v.T)
            Y_i = np.matmul(t, pilot[i].conj())
            Y[i] = Y_i
        shape = np.shape(Y)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    Y[i, j, k] = Y[i, j, k] + WirelessSystem.complex_Gaussian(upNoisePower)
        return Y

    def z(self):
        Y = self.pilot_signal_generator()
        shape = np.shape(Y)
        if self.requires_location:
            z = torch.zeros(torch.Size([self.user_num, 2 * shape[1] * shape[2] + 3]))
        else:
            z = torch.zeros(torch.Size([self.user_num, 2 * shape[1] * shape[2]]))
        for k in range(self.user_num):
            Y_k = Y[k]
            y_k = self.vectorize_2D(Y_k)
            z_k_imag = np.imag(y_k)
            z_k_real = np.real(y_k)
            z_k = np.concatenate([z_k_real, z_k_imag], 1)
            if self.requires_location:
                z[k, 0:2 * shape[1] * shape[2]] = torch.from_numpy(z_k)
                z[k, 2 * shape[1] * shape[2]] = self.user_array[k].xx
                z[k, 2 * shape[1] * shape[2] + 1] = self.user_array[k].yy
                z[k, 2 * shape[1] * shape[2] + 2] = self.user_array[k].zz
            else:
                z[k] = torch.from_numpy(z_k)
        z = Variable(z, requires_grad=True)
        return z


