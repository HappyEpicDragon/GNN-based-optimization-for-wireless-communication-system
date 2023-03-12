import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from BasicStuff import *


class System(nn.Module):
    def __init__(self, info, wireless_system):
        super(System, self).__init__()
        self.wireless_system = wireless_system
        self.GNN = GNN(info)

    def forward(self, Y):
        W, v = self.GNN(Y)
        v = torch.atleast_2d(v)
        shape = W.shape
        W_v = torch.zeros(torch.Size([shape[0] + 1, shape[1]]))
        W_v[0] = v
        W_v[1:shape[0] + 1] = W
        return W_v


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.f_nn = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.f_nn(z)


class Agg_Com(nn.Module):
    def __init__(self, irs=False, DNN_0=None, input_dim=0):
        super(Agg_Com, self).__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.IRS_DNN = DNN_0
        self.irs = irs
        self.NN = DNN(input_dim, input_dim)
        self.IS_IRS_DNN = DNN(input_dim * 2, input_dim)
        self.NOT_IRS_DNN = DNN(input_dim * 3, input_dim)

    def aggregate(self, Y, i):  # i start from 0
        shape = Y.shape
        if not self.irs:
            neighbor = torch.zeros(torch.Size([Y.shape[0] - 1, Y.shape[1]]))
            k = 0
            for j in range(Y.shape[0]):
                if i == j:
                    pass
                else:
                    neighbor[k, :] = Y[j, :]
                    k = k + 1
        else:
            neighbor = torch.zeros(torch.Size([Y.shape[0], Y.shape[1]]))
            for j in range(Y.shape[0]):
                neighbor[j] = Y[j]
        shape_nei = neighbor.shape
        temp = torch.clone(neighbor)
        temp = temp.to(self.DEVICE)
        for j in range(shape_nei[0]):
            neighbor[j] = self.NN(temp[j])
        if self.irs:
            temp = torch.clone(neighbor)
            neighbor = torch.unsqueeze(temp, 0)
            output = F.adaptive_avg_pool2d(neighbor, output_size=[1, shape[1]])
            temp = torch.clone(output)
            temp_1 = torch.squeeze(temp)
            output = torch.unsqueeze(temp_1, 0)
            return output
        else:
            temp = torch.clone(neighbor)
            neighbor = torch.unsqueeze(temp, 0)
            output = F.adaptive_max_pool2d(neighbor, output_size=[1, shape[1]])
            temp = torch.clone(output)
            temp_1 = torch.squeeze(temp)
            output = torch.unsqueeze(temp_1, 0)
            return output

    def combine(self, avg, Y, i):
        aggregate_output = self.aggregate(Y, i)
        shape = aggregate_output.shape
        INPUT = avg
        IRS_DNN_output = self.IRS_DNN(INPUT)
        if not self.irs:
            middle = torch.zeros(torch.Size([1, shape[1] * 3]))
            middle[0, 0:1 * shape[1]] = IRS_DNN_output
            middle[0, 1 * shape[1]:2 * shape[1]] = Y[i]
            middle[0, 2 * shape[1]:3 * shape[1]] = aggregate_output[0]
            middle = middle.to(self.DEVICE)
            NN_output = self.NOT_IRS_DNN(middle)
        else:
            middle = torch.zeros(torch.Size([1, shape[1] * 2]))
            middle[0, 0:1 * shape[1]] = IRS_DNN_output
            middle[0, 1 * shape[1]:2 * shape[1]] = aggregate_output
            middle = middle.to(self.DEVICE)
            NN_output = self.IS_IRS_DNN(middle)
        return NN_output

    def forward(self, avg, Y, i):
        output = self.combine(avg, Y, i)
        return output


class updating_layer(nn.Module):
    def __init__(self, user_num, IRS_DNN, input_dim):
        super(updating_layer, self).__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agg_com_NN = []
        for i in range(user_num + 1):
            if i == 0:
                is_IRS = True
            else:
                is_IRS = False
            self.agg_com_NN.append(
                Agg_Com(is_IRS, IRS_DNN, input_dim).to(self.DEVICE)
            )

    def forward(self, Y, avg):
        avg_0 = torch.clone(avg)
        avg = self.agg_com_NN[0](avg_0, Y, 0)
        temp = torch.clone(Y)
        for i in range(1, len(self.agg_com_NN)):
            Y[i - 1] = self.agg_com_NN[i](avg, temp, i)
        return Y, avg


class normalization_layer(nn.Module):
    def __init__(self, info):
        super(normalization_layer, self).__init__()
        self.threshold = np.exp(info.downlink_power / 10 * np.log(10)) / 1000

    def forward(self, Z, arg="W"):
        shape = Z.shape
        normal = torch.zeros(shape)
        if arg == "W":
            for i in range(shape[0]):
                norm = torch.norm(Z[i])
                normal[i] = Z[i] / norm * np.sqrt(self.threshold)
            return normal
        else:
            mi = np.int_(shape[1] / 2)
            out = torch.zeros([1, shape[1]])
            for i in range(mi):
                v_i = torch.zeros(torch.Size([1, 2]))
                v_i[0, 0] = Z[0, i]
                v_i[0, 1] = Z[0, i + mi]
                norm = torch.norm(v_i)
                temp = torch.clone(v_i)
                v_i = temp / norm
                out[0, i] = v_i[0, 0]
                out[0, i + mi] = v_i[0, 1]
        return out


class GNN(nn.Module):
    def __init__(self, info):
        super(GNN, self).__init__()
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.N = info.IRS_antennas
        self.user_nums = info.user_num
        self.updating_layer_num = info.updating_layer_num
        self.input_dim = 2 * info.pilot_length + 3
        self.input_DNNs = []
        for i in range(2):
            self.input_DNNs.append(
                DNN(self.input_dim, self.input_dim).to(self.DEVICE)
            )

        self.IRS_DNN = DNN(self.input_dim, self.input_dim).to(self.DEVICE)
        self.updating_layers = []
        for i in range(self.updating_layer_num):
            self.updating_layers.append(
                updating_layer(self.user_nums, self.IRS_DNN, self.input_dim).to(self.DEVICE)
            )
        self.linear_layers = []
        for i in range(2):
            self.linear_layers.append(
                nn.Linear(self.input_dim, self.N * 2).to(self.DEVICE)
            )
        self.W_normalize = normalization_layer(info).to(self.DEVICE)
        self.v_normalize = normalization_layer(info).to(self.DEVICE)

    def forward(self, Y):
        shape = Y.shape
        Y_3D = torch.unsqueeze(Y, 0)
        avg = F.adaptive_avg_pool2d(Y_3D, output_size=[1, shape[1]])
        temp = torch.clone(avg)
        temp_1 = torch.squeeze(temp)
        avg = torch.unsqueeze(temp_1, 0)
        temp = torch.clone(avg)
        avg = self.input_DNNs[0](temp)
        temp = torch.zeros(torch.Size([shape[0], shape[1]]))
        for i in range(shape[0]):
            temp[i] = self.input_DNNs[1](Y[i])
        Y = torch.clone(temp)

        for i in range(len(self.updating_layers)):
            Y_0, avg_0 = self.updating_layers[i](Y, avg)
            Y = torch.clone(Y_0)
            avg = torch.clone(avg_0)
        W_0 = torch.zeros(torch.Size([shape[0], self.N * 2]))
        v_0 = self.linear_layers[0](avg)
        Y = Y.to(self.DEVICE)
        for i in range(shape[0]):
            W_0[i] = self.linear_layers[1](Y[i])
        W = self.W_normalize(W_0, "W")
        v = self.v_normalize(v_0, 'v')
        W.retain_grad()
        v.retain_grad()
        return W, v


class minus_rate_function(nn.Module):
    def __init__(self, A, h_d, info):
        super(minus_rate_function, self).__init__()
        self.A = A
        self.h_d = h_d
        self.user_nums = info.user_num
        self.downlink_noise_power = info.downlink_noise_power

    def forward(self, W_v):
        shape = W_v.shape
        W = W_v[1:shape[0]]
        v = W_v[0]
        shape_W = W.shape
        shape_v = v.shape
        W_real = W[:, 0:np.int_(shape_W[1] / 2)]
        W_imag = W[:, np.int_(shape_W[1] / 2):shape_W[1]]
        A_real = np.real(self.A)
        A_imag = np.imag(self.A)
        h_d_real = np.real(self.h_d)
        h_d_imag = np.imag(self.h_d)
        v_real = v[0:np.int_(shape_v[0] / 2)]
        v_imag = v[np.int_(shape_v[0] / 2):shape_v[0]]
        n_0 = np.exp(self.downlink_noise_power / 10 * np.log(10)) / 1000
        W_matrix = []
        h_d_matrix = []
        A_matrix = []
        v_matrix = []
        for i in range(np.shape(W_real)[1]):
            W_real_i = torch.atleast_2d(W_real[:, i])
            W_imag_i = torch.atleast_2d(W_imag[:, i])
            matrix_1_1 = torch.zeros(torch.Size([W_imag_i.shape[0], W_imag_i.shape[1] * 2]))
            matrix_1_1[:, 0:W_imag_i.shape[1]] = W_real_i
            matrix_1_1[:, W_imag_i.shape[1]:W_imag_i.shape[1] * 2] = -W_imag_i
            matrix_1_2 = torch.zeros(torch.Size([W_imag_i.shape[0], W_imag_i.shape[1] * 2]))
            matrix_1_2[:, 0:W_imag_i.shape[1]] = W_imag_i
            matrix_1_2[:, W_imag_i.shape[1]:W_imag_i.shape[1] * 2] = W_real_i
            matrix_1 = torch.zeros(torch.Size([matrix_1_1.shape[0] * 2, matrix_1_1.shape[1]]))
            matrix_1[0:matrix_1_1.shape[0]] = matrix_1_1
            matrix_1[matrix_1_1.shape[0]:matrix_1_1.shape[0] * 2] = matrix_1_2
            W_matrix.append(matrix_1)

        for i in range(self.user_nums):
            h_d_real_i = np.array([h_d_real[i]])
            h_d_imag_i = np.array([h_d_imag[i]])
            matrix_2 = np.concatenate([h_d_real_i.T, h_d_imag_i.T], axis=0)
            h_d_matrix.append(matrix_2)

            A_real_i = np.array([A_real[i]])
            A_imag_i = np.array([A_imag[i]])
            matrix_3_1 = np.concatenate([A_real_i, -A_imag_i], axis=1)
            matrix_3_2 = np.concatenate([A_imag_i, A_real_i], axis=1)
            matrix_3 = np.concatenate([matrix_3_1, matrix_3_2], axis=2)
            A_matrix.append(matrix_3)

        v_real_i = torch.unsqueeze(v_real, 0)
        v_imag_i = torch.unsqueeze(v_imag, 0)
        matrix_4 = torch.zeros(torch.Size([v_imag_i.shape[1] * 2, v_imag_i.shape[0]]))
        matrix_4[0:v_imag_i.shape[1]] = v_real_i.T
        matrix_4[v_imag_i.shape[1]:v_imag_i.shape[1] * 2] = v_imag_i.T
        v_matrix.append(matrix_4)

        A_matrix = torch.Tensor(A_matrix)
        h_d_matrix = torch.Tensor(h_d_matrix)
        Sum = torch.zeros(1)
        Sum_log = torch.zeros(1)
        times = 0
        for k in range(self.user_nums):
            gamma = []
            for i in range(len(W_matrix)):
                a = torch.matmul(A_matrix[k][0], v_matrix[0])
                b = h_d_matrix[k] + a
                gamma_i = torch.matmul(W_matrix[i], b)
                gamma.append(gamma_i)

            gamma_sum = 0
            for i in range(len(gamma)):
                if i == k:
                    pass
                else:
                    norm = torch.norm(gamma[i])
                    temp = torch.clone(torch.Tensor([gamma_sum]))
                    gamma_sum = temp + norm

            R_k_log = torch.log(1 + torch.norm(gamma[k]) / (gamma_sum + n_0))
            R_k = torch.norm(gamma[k]) / (gamma_sum + n_0)
            temp = torch.clone(Sum)
            Sum = temp + R_k
            Sum_log = Sum_log + R_k_log
            temp = torch.clone(torch.Tensor([times]))
            times = temp + 1

        # expectation = Sum / times
        expectation = -Sum * 1e6
        print(Sum_log.detach().numpy())
        # expectation = - expectation * 10000
        return expectation
