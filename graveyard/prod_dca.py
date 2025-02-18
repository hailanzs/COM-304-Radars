from random import sample

from numpy.lib.function_base import unwrap
import mmwave as mm
from mmwave.dataloader import DCA1000, adc
import scipy
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io as sio
import os
from datetime import datetime
import radar
# import cmath

## TODO: run mmWave studio and Lua code in background using python

########################################################
########## 1843 CODE#############################
########################################################
def producer_real_time_1843(q, index, radar1):
    
    print('')
    print("num tx: ", radar1.num_tx)
    print("num rx: ", radar1.num_rx)
    print("samples_per_chirp: ", radar1.samples_per_chirp)
    print("chirp_loops: ", radar1.chirp_loops) 
    print("PERIODICITY: ", radar1.periodicity) 
    print("NUM_FRAMES: ", radar1.NUM_FRAMES) 
    second_p = 0
    second_p_plot = 0
    frm_ctr = 0
    total = np.zeros(shape=(radar1.NUM_FRAMES, radar1.samples_per_chirp), dtype=np.complex128)
    unwrapped_phase = np.zeros(shape=(radar1.NUM_FRAMES), dtype=np.complex128)
    freq_spec = 0
    dca = DCA1000()
    dca.sensor_config(chirps=radar1.num_tx, chirp_loops=radar1.chirp_loops, num_rx=radar1.num_rx, num_samples=radar1.samples_per_chirp)
    prev_time = time.time()
    while True:
        adc_data = dca.read()
        org_data = dca.organize(raw_frame=adc_data, num_chirps=radar1.num_tx*radar1.chirp_loops,
         num_rx=radar1.num_rx, num_samples=radar1.samples_per_chirp, num_frames=1, model='1843')
        frm_ctr+=1
        x = 0

        x = np.sum(org_data, axis=(0, 1))
        x -= np.mean(x, axis=-1, keepdims=True)
        fx = scipy.fft.fft(x, axis=-1)

        afx = np.squeeze(np.abs(fx))

        total[0:-radar1.chirp_loops, :] = total[radar1.chirp_loops:, :]
        total[-radar1.chirp_loops:, :] = fx
        max_idx = np.argmax(np.sum(np.abs(total[3:]),axis=0))

        now = time.time()
        if now - prev_time > 0.1:
            q.put(["fft", afx])
            prev_time = now
