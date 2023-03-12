from Parameters import *
from BCD import *


def LMMSE_main(system_info):
    h_d = wireless_system.h_d
    h_r = wireless_system.h_r
    G = wireless_system.G
    s = system_info.downlink_noise_power
    sigma2 = np.exp(s / 10 * np.log(10)) / 1000
    P_T = system_info.downlink_power
    P_T = np.exp(P_T / 10 * np.log(10)) / 1000

