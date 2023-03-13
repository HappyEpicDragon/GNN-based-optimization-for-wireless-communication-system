from Parameters import *
from BCD import *


class LMMSE_channel_estimation:
    def __init__(self, system):
        self.system = system
        self.h_d, self.G, self.h_r = self.channel_estimation()

    def channel_estimation(self):
        A = self.system.A
        h_d = None
        G = None
        h_r = None
        return h_d, G, h_r


def LMMSE_main(system_info):
    LMMSE = LMMSE_channel_estimation(wireless_system)
    h_d = LMMSE.h_d
    h_r = LMMSE.h_r
    G = LMMSE.G
    s = system_info.downlink_noise_power
    sigma2 = np.exp(s / 10 * np.log(10)) / 1000
    P_T = system_info.downlink_power
    P_T = np.exp(P_T / 10 * np.log(10)) / 1000
    bcd = BCD(h_d, G, h_r, sigma2, P_T, wireless_system.v, system_info)
    bcd.optimize()

