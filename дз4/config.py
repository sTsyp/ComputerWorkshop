import numpy as np
w = 1920
h = 1080
N = 5000
dt = 0.1
asp = w / h
perception = 1 / 20
better_walls_w = 0.05
vrange = (0, 0.1)
arange = (0, 0.01)
cnt_rely_on = 50
frame = 0
coeffs = np.array([0.9, 0.2, 0.8, 0.05])
