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

FLUXRAMP_STEP = 5e-7 # A
FLUXRAMP_STEPTIME = 0.05 # s
SETTLING_TIME = 0.1 # s
FLUX_MAXJUMP = 2e-4 # A

#%%
# Pump source

from RsInstrument import RsInstrument

rfsource = RsInstrument('TCPIP::169.254.2.22::INSTR', id_query=True, reset=False)
rfsource.visa_timeout = 50000000
rfsource.opc_timeout = 1000000
rfsource.instrument_status_checking = True
rfsource.opc_query_after_write = True
rfsource.write_str_with_opc(":output off")
SETTLING_TIME = 0.1  # seconds

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

filename = '{datetime}_qm_JPA_pump_power_vs_pump_freq'
fpath = data_path(filename, datesuffix='_qm')

Iflux = -0.054e-3 # A
fsignal = config.vnaLO + config.vnaIF

Navg = 100
fpump = np.arange(9e9, 13e9, 15e6)
Ppump = np.arange(-30, 11, 1)

# Ramp to flux
assert np.abs(Iflux) < 1e-3 # Limit 1mA thermocoax
print(f"Setting flux current ({abs(fluxbias.current()-Iflux)/FLUXRAMP_STEP*FLUXRAMP_STEPTIME/60:.1f}min)")
fluxbias.ramp_current(Iflux, FLUXRAMP_STEP, FLUXRAMP_STEPTIME)
time.sleep(2) # settle

# Define program using settings above
with qua.program() as vna:
    nPpump = qua.declare(int)
    nfpump = qua.declare(int)
    n = qua.declare(int)
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    qua.pause()
    # First run for reference line
    with qua.for_(n, 0, n < Navg, n + 1):
        # qua.wait(config.cooldown_clk, 'vna') # not really necessary, VNA is CW measurement
        qua.wait(rand.rand_int(50)+4, 'vna')
        qua.measure('readout', 'vna', None,
                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
        qua.save(I, I_st)
        qua.save(Q, Q_st)

    with qua.for_(nPpump, 0, nPpump < Ppump.size, nPpump + 1):
        with qua.for_(nfpump, 0, nfpump < fpump.size, nfpump + 1):
            qua.pause()
            with qua.for_(n, 0, n < Navg, n + 1):
                # qua.wait(config.cooldown_clk, 'vna') # not really necessary, VNA is CW measurement
                qua.wait(rand.rand_int(50)+4, 'vna')
                qua.measure('readout', 'vna', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)

    with qua.stream_processing():
        I_st.buffer(Navg).map(qua.FUNCTIONS.average(0)).save_all('I')
        Q_st.buffer(Navg).map(qua.FUNCTIONS.average(0)).save_all('Q')

# Start QM program
QMSLEEP = 0.05
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_vna(qm, config)
job = qm.execute(vna)

res_handles = job.result_handles
I_handle = res_handles.get('I')
Q_handle = res_handles.get('Q')
while not job.is_paused():
    mpl_pause(QMSLEEP)

# Acquire reference, Pump off
rfsource.write_str_with_opc(":output off")
job.resume()
while not job.is_paused() and not job.status == 'completed':
    mpl_pause(QMSLEEP)


refsignal = np.nan+0j
dataS21 = np.full((Ppump.size, fpump.size), np.nan+0j)
tracetime = np.full((Ppump.size, fpump.size), np.nan)

# Live plotting
from matplotlib.colors import CenteredNorm
fig, ax = plt.subplots()
img = plt2dimg(ax, fpump/1e9, Ppump, 20*np.log10(np.abs(dataS21.T/refsignal)), norm=CenteredNorm(), cmap='coolwarm')
fig.colorbar(img, label="Gain  S/Sref  [dB]")
ax.set_ylabel('Pump power / dBm')
ax.set_xlabel('Pump freq / GHz')
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
title = (
    f"signal {config.vnaLO/1e9:.5f}GHz+{config.vnaIF/1e6:.3f}MHz   Navg {Navg}"
    f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.vna_output_gain:+.1f}dB"
    f",   {config.input_gain:+.1f}dB input gain"
    f"\n Iflux {Iflux*1e3:.4f}mA")
fig.suptitle(title, fontsize=10)
fig.show()


rfsource.write_str_with_opc(f':source:power {Ppump[0]:f}')
rfsource.write_str_with_opc(f':source:freq {fpump[0]/1e9:f}ghz')
rfsource.write_str_with_opc(":output on")
estimator = DurationEstimator(Ppump.size*fpump.size)
try:
    for i in range(Ppump.size):
        rfsource.write_str_with_opc(f":source:power {Ppump[i]:f}")
        for j in range(fpump.size):
            rfsource.write_str_with_opc(f':source:freq {fpump[j]/1e9:f}ghz')

            # Acquire a trace
            tracetime[i,j] = time.time()
            job.resume()
            while not job.is_paused() and not job.status == 'completed':
                mpl_pause(QMSLEEP)

            if (i*fpump.size + j) % 10 == 0:
                estimator.step(i*fpump.size + j)

        # Live plotting
        while I_handle.count_so_far() < (i+1)*fpump.size+1 or Q_handle.count_so_far() < (i+1)*fpump.size+1:
            mpl_pause(QMSLEEP)
        I, Q = I_handle.fetch_all()['value'], Q_handle.fetch_all()['value']
        refsignal = I[0] + 1j * Q[0]
        dataS21[:i+1] = (I[1:] + 1j * Q[1:]).reshape(i+1, fpump.size)
        plt2dimg_update(img, 20*np.log10(np.abs(dataS21/refsignal)).T)
finally:
    job.halt()
    estimator.end()
    try: # in case of interrupt
        I_handle.wait_for_all_values()
        Q_handle.wait_for_all_values()
        I, Q = I_handle.fetch_all()['value'], Q_handle.fetch_all()['value']
        refsignal = I[0] + 1j * Q[0]
        dataS21 = (I[1:] + 1j * Q[1:]).reshape(Ppump.size, fpump.size)
    except Exception as e:
        print(repr(e))
    np.savez_compressed(
        fpath, Ppump=Ppump, fpump=fpump, Navg=Navg,
        refsignal=refsignal, dataS21=dataS21, tracetime=tracetime,
        Iflux=Iflux, fsignal=fsignal,
        config=config.meta)
    print("Time per trace:", (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime)), 's')

    plt2dimg_update(img, 20*np.log10(np.abs(dataS21/refsignal)).T)
    fig.tight_layout()
    fig.savefig(fpath+'.png', dpi=300)

#%%
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
