
# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# Use this script to determine
# - electrical delay correction (config.PHASE_CORR)
# - resonator IF

import importlib
import numpy as np
import matplotlib.pyplot as plt
import qm.qua as qua
from qualang_tools.loops import from_array

from helpers import data_path, mpl_pause

import configuration_novna as config
import qminit

qmm = qminit.connect()

#%%

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_resonator_spec_ffttest'
fpath = data_path(filename, datesuffix='_qm')

Navg = 2000
freqs = np.arange(202e6, 212e6, 0.1e6)

with qua.program() as resonator_spec:
    n = qua.declare(int)  # variable for average loop
    n_st = qua.declare_stream()  # stream for 'n'
    f = qua.declare(int)  # variable to sweep freqs
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    I_st = qua.declare_stream()  # stream for I
    Q_st = qua.declare_stream()  # stream for Q
    rand = qua.lib.Random()

    with qua.for_(*from_array(f, freqs)):
        qua.update_frequency('resonator', f)
        with qua.for_(n, 0, n < Navg, n + 1):
            qua.reset_phase('resonator')
            #qua.wait(config.cooldown_clk, 'resonator')  # wait for resonator to decay
            #qua.wait(rand.rand_int(50)+4, 'resonator')
            #play('saturation', 'qubit')
            # qua.align()
            qua.measure('readout', 'resonator', None,
                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
        qua.save(n, n_st)

    with qua.stream_processing():
        I_st.with_timestamps().save_all('I')
        Q_st.with_timestamps().save_all('Q')

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
job = qm.execute(resonator_spec)

job.result_handles.wait_for_all_values()
I, Q = job.result_handles.get('I').fetch_all()['value'], job.result_handles.get('Q').fetch_all()['value']
t = job.result_handles.get('I').fetch_all()['timestamp']
Z = (I + 1j*Q).reshape(freqs.size, Navg)
np.savez_compressed(fpath, Z=Z, N=Navg, t=t, config=config.meta)

#%%
# Filtering

dt = np.median(np.diff(t))
print("Sample spacing", dt/1e3, "us")

tshaped = t.reshape(freqs.size, Navg)
print("Measurement length per freq point:", np.median(tshaped[:,-1]-tshaped[:,0])/1e6, "ms")

Z = (I + 1j*Q).reshape(freqs.size, Navg)
meanZ = np.mean(Z, axis=-1)
FZ = np.fft.fftshift(np.fft.fft(Z), axes=-1)
Ff = np.fft.fftshift(np.fft.fftfreq(Navg, d=dt*1e-9))

fidx = np.argmin(np.abs(Ff - 0))
FZ2 = FZ.copy()
FZ2[:,:fidx] = 0
FZ2[:,fidx+1:] = 0

Zinv = np.fft.ifft(np.fft.ifftshift(FZ, axes=-1), axis=-1)
Zfilt = np.fft.ifft(np.fft.ifftshift(FZ2, axes=-1), axis=-1)

fig, axs = plt.subplots(nrows=3, sharex=True)
axs[0].plot(freqs/1e6, np.abs(meanZ))
axs[1].pcolormesh(freqs/1e6, Ff/1e3, 20*np.log10(np.abs(FZ).T), antialiased=True)
axs[2].plot(freqs/1e6, np.abs(np.mean(Zinv, axis=-1)), label="Same signal")
axs[2].plot(freqs/1e6, np.abs(np.mean(Zfilt, axis=-1)), '--', label="Filtered, averaged")
axs[1].set_ylabel('f / kHz')
axs[-1].set_xlabel('IF / MHz')

#%%
# Selecting

dt = np.median(np.diff(t))
print("Sample spacing", dt/1e3, "us")
tshaped = t.reshape(freqs.size, Navg)
print("Measurement length per freq point:", np.median(tshaped[:,-1]-tshaped[:,0])/1e6, "ms")

# 50Hz trigger reference
period = 20e-3 * 1e9 # ns
phases = np.array([0, np.pi/2, np.pi, 2*3/4*np.pi]) # rad
maxdist = 100e-6 * 1e9 # ns

Z = (I + 1j*Q).reshape(freqs.size, Navg)
reft = np.arange(0, t[-1], period) # ns
delays = phases / (2*np.pi) * period # ns
Zsub = np.full((phases.size, freqs.size), np.nan+0j)
for j in range(phases.size):
    for i in range(freqs.size):
        # distance to a reference point
        dist = np.min(np.abs(tshaped[i,:,None] - (reft[None,:] + delays[j])), axis=1)
        mask = dist < maxdist
        Zsub[j,i] = np.mean(Z[i,mask])

Z = (I + 1j*Q).reshape(freqs.size, Navg)
meanZ = np.mean(Z, axis=-1)

fig, axs = plt.subplots(nrows=2, sharex=True)
axs[1].pcolormesh(freqs/1e6, Ff/1e3, 20*np.log10(np.abs(FZ).T))
axs[0].plot(freqs/1e6, np.abs(meanZ), 'k--', label="Full average")
for j in range(phases.size):
    axs[0].plot(freqs/1e6, np.abs(Zsub[j]), color=plt.cm.rainbow(j/phases.size), label=f"{delays[j]/1e6:.2f}ms delay")
axs[0].legend(fontsize=8)
axs[0].set_ylabel('|Z|')
axs[1].set_ylabel('f / kHz')
axs[-1].set_xlabel('IF / MHz')
fig.suptitle(f"Subselect every {period/1e6:.2f}ms samples.\n{Navg} samples per freq point", fontsize=10)
fig.tight_layout()
fig.savefig(fpath+'.png')

#%%
# IQ plane

plt.figure()
for k in range(freqs.size)[::3]:
    plt.plot(Zsub[:,k].real, Zsub[:,k].imag, 'C0.-')
plt.axis('equal')
plt.grid()
