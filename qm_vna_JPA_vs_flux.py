
# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# Use this script to determine
# - electrical delay correction (config.PHASE_CORR)
# - vna IF

import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import qm.qua as qua
from qualang_tools.loops import from_array

from helpers import data_path, mpl_pause, plt2dimg, plt2dimg_update, DurationEstimator

import configuration as config
import qminit

qmm = qminit.connect()

#%%

from qcodes.instrument_drivers.yokogawa.GS200 import GS200

try:
    fluxbias.close()
    pass
except: pass

fluxbias = GS200("source", 'TCPIP0::169.254.0.2::inst0::INSTR', terminator="\n")
assert fluxbias.source_mode() == 'CURR'
assert fluxbias.output() == 'on'

# min step 1e-5 mA = 1e-8 A
FLUXRAMP_STEP = 5e-8 # A
FLUXRAMP_STEPTIME = 0.05 # s
SETTLING_TIME = 0.1 # s
FLUX_MAXJUMP = 2e-4 # A

#%% Calibration
        
importlib.reload(config)
importlib.reload(qminit)

print("Running calibration...")
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_vna(qm, config)
cal = qm.octave.calibrate_element('vna', [(config.vnaLO, config.vnaIF)])
# Note: You need to reopen qm to apply calibration settings

## Run output continuously for checking in the spectrum analyser
# with qua.program() as mixer_cal_vna:
#     #qua.update_frequency('qubit', -385e6)
#     with qua.infinite_loop_():
#         qua.play('const', 'vna')
# print("Playing constant pulse on qubit channel...")
# qm = qmm.open_qm(config.qmconfig)
# qminit.octave_setup_vna(qm, config)
# job = qm.execute(mixer_cal_vna)

#%%

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_vna_vs_flux_test'
fpath = data_path(filename, datesuffix='_qm')

Iflux = np.concatenate([np.linspace(0.15e-3, 0.1e-3, 201)])
I_step = np.mean(np.abs(np.diff(Iflux)))
print(f"Iflux measurement step: {I_step*1e3:.5f}mA avg")
Nflux = Iflux.size

assert np.all(np.abs(Iflux) < 1e-3) # Limit 1mA thermocoax

Navg = 100

freqs = np.arange(-402e6, 402e6, 4e6)
Nf = len(freqs)

with qua.program() as vna:
    nflux = qua.declare(int) # number of flux values
    n = qua.declare(int)  # variable for average loop
    f = qua.declare(int)  # variable to sweep freqs
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    n_st = qua.declare_stream()  # stream for 'n'
    I_st = qua.declare_stream()  # stream for I
    Q_st = qua.declare_stream()  # stream for Q
    rand = qua.lib.Random()

    with qua.for_(nflux, 0, nflux < Nflux, nflux + 1):
        qua.pause()
        with qua.for_(n, 0, n < Navg, n + 1):
            with qua.for_(*from_array(f, freqs)):  # Notice it's <= to include f_max (This is only for integers!)
                qua.update_frequency('vna', f)  # update frequency of vna element
                qua.wait(config.cooldown_clk, 'vna')  # wait for resonator to decay
                qua.wait(rand.rand_int(50)+4, 'vna')
                qua.measure('readout', 'vna', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)
            qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(Navg, len(freqs)).map(qua.FUNCTIONS.average(0)).save_all('I')
        Q_st.buffer(Navg, len(freqs)).map(qua.FUNCTIONS.average(0)).save_all('Q')


print(f"Setting flux current ({abs(fluxbias.current()-Iflux[0])/FLUXRAMP_STEP*FLUXRAMP_STEPTIME/60:.1f}min)")
fluxbias.ramp_current(Iflux[0], FLUXRAMP_STEP, FLUXRAMP_STEPTIME)
print("Wait for flux current to settle")
time.sleep(2)

QMSLEEP = 0.05
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_vna(qm, config)
job = qm.execute(vna)

res_handles = job.result_handles
I_handle = res_handles.get('I')
Q_handle = res_handles.get('Q')
while not job.is_paused():
    mpl_pause(QMSLEEP)

dataS21 = np.full((Nflux, Nf), np.nan+0j)
tracetime = np.full(Nflux, np.nan)
tflux = []
tqm = []


# Live plotting
fig, ax = plt.subplots()
img = plt2dimg(ax, (config.vnaLO+freqs)/1e9, Iflux*1e3, np.angle(dataS21).T, vmin=-np.pi, vmax=np.pi, cmap='twilight')
fig.colorbar(img, label="arg S")
ax.set_ylabel('Current / mA')
ax.set_xlabel('Freq / GHz')
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
title = (
    f"LO={config.vnaLO/1e9:.5f}GHz   Navg {Navg}"
    f"   electric delay {config.VNA_PHASE_CORR:.3e}rad/Hz"
    f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.vna_output_gain:+.1f}dB"
    f",   {config.input_gain:+.1f}dB input gain")
fig.suptitle(title, fontsize=10)
fig.show()


estimator = DurationEstimator(Nflux)
try:
    for i in range(Nflux):
        tstart = time.time()
        fluxbias.ramp_current(Iflux[i], FLUXRAMP_STEP, FLUXRAMP_STEPTIME)
        tflux.append(time.time()-tstart)
        tracetime[i] = time.time()

        tstart = time.time()
        job.resume()
        while not job.is_paused() and not job.status == 'completed':
            mpl_pause(QMSLEEP)
        tqm.append(time.time()-tstart)

        if i%1 == 0 or i == Nflux-1:
            I = I_handle.fetch_all()['value']
            Q = Q_handle.fetch_all()['value']
            l = min(I.shape[0], Q.shape[0])
            dataS21[:l] = I[:l] + 1j * Q[:l]
            plt2dimg_update(img, np.angle(dataS21*np.exp(1j*freqs*config.VNA_PHASE_CORR)[None,:]).T, rescalev=False)

        estimator.step(i)
finally:
    job.halt()
    estimator.end()
    try: # in case of interrupt
        I = I_handle.fetch_all()['value']
        Q = Q_handle.fetch_all()['value']
        l = min(I.shape[0], Q.shape[0])
        dataS21[:l] = I[:l] + 1j * Q[:l]
    except:
        pass
    np.savez_compressed(
        fpath, Navg=Navg, f=freqs, dataS21=dataS21, Iflux=Iflux,
        config=config.meta)
    print("Time per trace:", (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime)), 's')
    print("Time for flux set:", np.mean(tflux), "s")
    print("Time for QM execution:", np.mean(tqm), "s")

    plt2dimg_update(img, np.angle(dataS21*np.exp(1j*freqs*config.VNA_PHASE_CORR)[None,:]).T, rescalev=False)
    fig.tight_layout()
    fig.savefig(fpath+'.png', dpi=300)

    f = freqs + config.vnaLO
    S21 = dataS21 * 2**12 / config.readout_len * np.exp(1j*freqs*config.VNA_PHASE_CORR)[None,:]
    S21change = S21 / S21[0]
    absS = 20*np.log10(np.abs(dataS21))
    argS = np.unwrap(np.unwrap(np.angle(dataS21), axis=0))
    argS = np.unwrap(np.unwrap(np.angle(dataS21*np.exp(1j*freqs*config.VNA_PHASE_CORR)[None,:]), axis=0))
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, layout='constrained')
    im = axs[0,0].pcolormesh(f/1e9, Iflux*1e3, 20*np.log10(np.abs(S21)))
    fig.colorbar(im, ax=axs[0,0], label="|S| / dB")

    im = axs[0,1].pcolormesh(f/1e9, Iflux*1e3, 20*np.log10(np.abs(S21change)))
    fig.colorbar(im, ax=axs[0,1], label="|S/S_0| / dB")
    
    im = axs[1,0].pcolormesh(f/1e9, Iflux*1e3, np.unwrap(np.unwrap(np.angle(S21), axis=0)))
    fig.colorbar(im, ax=axs[1,0], label="arg S")
    
    im = axs[1,1].pcolormesh(f/1e9, Iflux*1e3, np.unwrap(np.unwrap(np.angle(S21change), axis=0)))
    fig.colorbar(im, ax=axs[1,1], label="arg(S/S_0)")
    
    axs[0,0].set_ylabel("Flux current / mA")
    axs[-1,0].set_xlabel("Probe Freq / GHz")
    fig.suptitle(title, fontsize=10)
    fig.savefig(fpath+'_all.png', dpi=300)

#%%

config.VNA_PHASE_CORR = 1.23e-6
plt.figure()
plt.plot(freqs, np.unwrap(np.angle(dataS21[0]*np.exp(1j*freqs*config.VNA_PHASE_CORR))))

#%%

f = freqs + config.vnaLO
absS = 20*np.log10(np.abs(dataS21))
argS = np.unwrap(np.unwrap(np.angle(dataS21*np.exp(1j*freqs*config.VNA_PHASE_CORR)[None,:]), axis=0))

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, layout='constrained')
im = axs[0,0].pcolormesh(f/1e9, Iflux*1e3, absS, shading='auto')
fig.colorbar(im, ax=axs[0,0], label="|S| / dB")

axs[0,1].pcolormesh(f/1e9, Iflux*1e3, (absS-absS[0][None,:]))
fig.colorbar(im, ax=axs[0,1], label="|S| / dB - first line")

axs[1,0].pcolormesh(f/1e9, Iflux*1e3, argS)
fig.colorbar(im, ax=axs[1,0], label="arg S")

axs[1,1].pcolormesh(f/1e9, Iflux*1e3, (argS-argS[0][None,:]))
fig.colorbar(im, ax=axs[1,1], label="arg S - first line")

axs[0,0].set_ylabel("Flux current / mA")
axs[-1,0].set_xlabel("Probe Freq / GHz")
