from BasicStuff import *

info = info_container(user_num=3,
                      positions=[[100, -100, 0],
                                 [0, 0, 0],
                                 [35, -35, -20],
                                 [35, 35, -20],
                                 [5, 35, -20]],
                      BS_antennas=8,
                      IRS_antennas=100,
                      pilot_length=120,
                      updating_layer_num=3,
                      uplink_power=15,
                      downlink_power=20,
                      uplink_noise_power=-100,
                      downlink_noise_power=-85)

wireless_system = WirelessSystem(info)


