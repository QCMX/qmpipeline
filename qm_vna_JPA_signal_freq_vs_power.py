import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import qm.qua as qua
import qm.octave as octave
from qualang_tools.loops import from_array, get_equivalent_log_array

from helpers import data_path, mpl_pause, plt2dimg, plt2dimg_update, DurationEstimator

import configuration_vna as config
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

filename = '{datetime}_qm_JPA_signal_power_vs_freq'
fpath = data_path(filename, datesuffix='_qm')

Iflux = -0.025e-3 # A
fpump = 10.481e9
Ppump = -12.7

ifsignal = np.arange(100e6, 300e6, 5e6)
Psignal = np.arange(-60, 0.1, 5)

Navg = 100

assert config.readout_amp == 0.316
amps = 10**(Psignal/20)
assert np.all(amps <= 1)

# Ramp to flux
assert np.abs(Iflux) < 1e-3 # Limit 1mA thermocoax
if fluxbias.current() != Iflux:
    print("Hysteresis, start from Iflux=0mA")
    fluxbias.ramp_current(0, FLUXRAMP_STEP, FLUXRAMP_STEPTIME)
    print(f"Setting flux current ({abs(fluxbias.current()-Iflux)/FLUXRAMP_STEP*FLUXRAMP_STEPTIME/60:.1f}min)")
    fluxbias.ramp_current(Iflux, FLUXRAMP_STEP, FLUXRAMP_STEPTIME)
    time.sleep(2) # settle

# Define program using settings above
with qua.program() as vna:
    a = qua.declare(qua.fixed)
    f = qua.declare(int)
    n = qua.declare(int)
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    a_st = qua.declare_stream()
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    with qua.for_each_(a, amps):
        qua.save(a, a_st)
        with qua.for_(n, 0, n < Navg, n + 1):
            with qua.for_(*from_array(f, ifsignal)):
                #qua.wait(config.cooldown_clk, 'vna')
                qua.wait(rand.rand_int(50)+4, 'vna')
                qua.update_frequency('vna', f)
                qua.measure('readout'*qua.amp(a), 'vna', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)

    with qua.stream_processing():
        a_st.save('a')
        Iavg = I_st.buffer(Navg, ifsignal.size).map(qua.FUNCTIONS.average(0))
        Qavg = Q_st.buffer(Navg, ifsignal.size).map(qua.FUNCTIONS.average(0))
        Iavg.save_all('I')
        Qavg.save_all('Q')
        ((I_st*I_st).buffer(Navg, ifsignal.size).map(qua.FUNCTIONS.average(0)) - Iavg*Iavg).save_all('Ivar')
        ((I_st*I_st).buffer(Navg, ifsignal.size).map(qua.FUNCTIONS.average(0)) - Qavg*Qavg).save_all('Qvar')


def acquire():
    qm = qmm.open_qm(config.qmconfig)
    qminit.octave_setup_vna(qm, config)
    job = qm.execute(vna)
    
    lasta = None
    while job.result_handles.is_processing():
        newa = job.result_handles.get('a').fetch_all()
        if newa != lasta:
            print(newa)
            lasta = newa
        mpl_pause(0.1)
    
    job.result_handles.wait_for_all_values()
    I = job.result_handles.get('I').fetch_all()['value']
    Q = job.result_handles.get('Q').fetch_all()['value']
    Ivar = job.result_handles.get('Ivar').fetch_all()['value']
    Qvar = job.result_handles.get('Qvar').fetch_all()['value']
    return I + 1j*Q, Ivar + 1j*Qvar

print("Acquiring ref...")
rfsource.write_str_with_opc(":output off")

Zofee, Zofeevar = acquire()

print("Acquiring signal...")
rfsource.write_str_with_opc(f':source:power {Ppump:f}')
rfsource.write_str_with_opc(f':source:freq {fpump/1e9:f}ghz')
rfsource.write_str_with_opc(":output on")

Zon, Zonvar = acquire()

print("Saving")
np.savez_compressed(
    fpath, Zon=Zon, Zoff=Zoff, Zonvar=Zonvar, Zoffvar=Zoffvar,
    ifsignal=ifsignal, Psignal=Psignal, amps=amps,
    Iflux=Iflux, Ppump=Ppump, fpump=fpump, Navg=Navg,
    tracetime=time.time(),
    config=config.meta)

# Shut down
rfsource.write_str_with_opc(":output off")
qm.octave.set_rf_output_mode('vna', octave.RFOutputMode.off)

# Plot
from matplotlib.colors import CenteredNorm
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, layout='constrained')
im = plt2dimg(axs[0,0], (ifsignal+config.vnaLO)/1e9, Psignal, 20*np.log10(Zon))
fig.colorbar(im, ax=axs[0,0], orientation='horizontal', shrink=0.8, label='|S| / dB')
im = plt2dimg(axs[0,1], (ifsignal+config.vnaLO)/1e9, Psignal, 20*np.log10(Zon/Zoff), norm=CenteredNorm(), cmap='coolwarm')
fig.colorbar(im, ax=axs[0,1], orientation='horizontal', shrink=0.8, label='Gain  |Son/Soff| / dB')
snron = np.abs(Zon) / np.sqrt(Zonvar.real + Zonvar.imag)
snroff = np.abs(Zon) / np.sqrt(Zonvar.real + Zonvar.imag)
im = plt2dimg(axs[1,0], (ifsignal+config.vnaLO)/1e9, Psignal, snron)
fig.colorbar(im, ax=axs[1,0], orientation='horizontal', shrink=0.8, label='|S| / dB')
im = plt2dimg(axs[1,1], (ifsignal+config.vnaLO)/1e9, Psignal, 10*np.log10(snron/snroff), norm=CenteredNorm(), cmap='coolwarm')
fig.colorbar(im, ax=axs[1,1], orientation='horizontal', shrink=0.8, label='SNR gain: 10*np.log10(SNRon / SNR off|)')
axs[0,0].set_ylabel(f"Signal power / dBm  {config.vna_output_gain:+.1f}dB")
axs[-1,0].set_xlabel("Signal freq / GHz")
title = (
    f"signal LO {config.vnaLO/1e9:.5f}GHz   Navg {Navg}"
    f"\n{config.readout_len}ns readout, {config.input_gain:+.1f}dB input gain"
    f"\nIflux {Iflux*1e3:.4f}mA  Pump {fpump/1e9:.5f}GHz {Ppump:.1f}dBm")
fig.suptitle(title, fontsize=10)
fig.savefig(fpath, dpi=300)
