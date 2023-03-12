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
    