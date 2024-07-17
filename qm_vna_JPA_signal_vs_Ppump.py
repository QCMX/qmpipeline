
# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# Use this script to determine
# - electrical delay correction (config.PHASE_CORR)
# - vna IF

import importlib
import numpy as np
import matplotlib.pyplot as plt
import qm.qua as qua
import qm.octave as octave
from qualang_tools.loops import from_array

from helpers import data_path, mpl_pause

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

FLUXRAMP_STEP = 5e-8 # A
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
 
print("Running calibration on VNA channel...")
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

filename = '{datetime}_qm_JPA_signal_vs_Ppump'
fpath = data_path(filename, datesuffix='_qm')

ifs = np.arange(-402e6, 402e6, 1e6)
Navg = 200

Ppump = np.arange(-17, -9.1, 1)

Iflux = fluxbias.current()
pumpmeta = {
    'f': float(rfsource.query(':source:freq?')),
    # 'power': float(rfsource.query(':source:power?')),
}
fs = config.vnaLO + ifs


with qua.program() as vna:
    nP = qua.declare(int)
    n = qua.declare(int)
    f = qua.declare(int)
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    # Pump off
    qua.pause()
    with qua.for_(n, 0, n < Navg, n + 1):
        with qua.for_(*from_array(f, ifs)):  # Notice it's <= to include f_max (This is only for integers!)
            qua.update_frequency('vna', f)  # update frequency of vna element
            # qua.wait(config.cooldown_clk, 'vna')  # wait for resonator to decay
            qua.wait(rand.rand_int(50)+4, 'vna')
            qua.measure('readout', 'vna', None,
                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)

    # Pump on
    with qua.for_(nP, 0, nP < Ppump.size, nP + 1):
        qua.pause()
        with qua.for_(n, 0, n < Navg, n + 1):
            with qua.for_(*from_array(f, ifs)):  # Notice it's <= to include f_max (This is only for integers!)
                qua.update_frequency('vna', f)  # update frequency of vna element
                # qua.wait(config.cooldown_clk, 'vna')  # wait for resonator to decay
                qua.wait(rand.rand_int(50)+4, 'vna')
                qua.measure('readout', 'vna', None,
                        qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)

    with qua.stream_processing():
        Iavg = I_st.buffer(Navg, len(ifs)).map(qua.FUNCTIONS.average(0))
        Qavg = Q_st.buffer(Navg, len(ifs)).map(qua.FUNCTIONS.average(0))
        Iavg.save_all('I')
        Qavg.save_all('Q')
        ((I_st*I_st).buffer(Navg, len(ifs)).map(qua.FUNCTIONS.average(0)) - Iavg*Iavg).save_all("Ivar")
        ((Q_st*Q_st).buffer(Navg, len(ifs)).map(qua.FUNCTIONS.average(0)) - Qavg*Qavg).save_all("Qvar")

# Start program
QMSLEEP = 0.05
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_vna(qm, config)
job = qm.execute(vna)
while not job.is_paused():
    mpl_pause(QMSLEEP)

print("Acquire reference, pump off")
rfsource.write_str_with_opc(":output off")
job.resume()
while not job.is_paused():
    mpl_pause(QMSLEEP)

print("Acquire signal, pump on")
rfsource.write_str_with_opc(f':source:power {Ppump[0]:f}')
rfsource.write_str_with_opc(":output on")
for i, P in enumerate(Ppump):
    print(f"{i+1}/{len(Ppump)}: {P:+.1f}dBm")
    rfsource.write_str_with_opc(f':source:power {Ppump[i]:f}')
    job.resume()
    while not job.is_paused() and job.status != 'completed':
        mpl_pause(QMSLEEP)

rfsource.write_str_with_opc(":output off")
qm.octave.set_rf_output_mode('vna', octave.RFOutputMode.off)

job.result_handles.wait_for_all_values()
I = job.result_handles.get('I').fetch_all()['value']
Q = job.result_handles.get('Q').fetch_all()['value']
Ivar = job.result_handles.get('Ivar').fetch_all()['value']
Qvar = job.result_handles.get('Qvar').fetch_all()['value']
Zraw = I + 1j*Q
Z = Zraw * np.exp(1j * ifs[None,:] * config.VNA_PHASE_CORR) / config.readout_len * 2**12
Zvar = (Ivar + 1j*Qvar) / config.readout_len * 2**12

S = Z[1:] / Z[0]
snr = np.abs(Z) / np.sqrt(np.abs(Zvar.real + Zvar.imag))

# Save
np.savez_compressed(
    fpath, Navg=Navg, ifs=ifs, fs=fs, Zraw=Zraw, Z=Z, Zvar=Zvar,
    S=S, snr=snr,
    pumpmeta=pumpmeta, Iflux=Iflux,
    config=config.meta)

# Plot
fig, axs = plt.subplots(nrows=4, sharex=True)
axs[2].plot(fs/1e9, 20*np.log10(np.abs(Z[0])), color=plt.cm.rainbow(0), label="Pump off")
for i, P in enumerate(Ppump):
    c = plt.cm.rainbow((i+1)/len(Ppump))
    axs[0].plot(fs/1e9, 20*np.log10(np.abs(S[i])), color=c)
    axs[1].plot(fs/1e9, np.unwrap(np.angle(S[i])), color=c, label=f"P={P:+.1f}dBm")
    axs[2].plot(fs/1e9, 20*np.log10(np.abs(Z[i+1])), color=c)
    axs[3].plot(fs/1e9, 20*np.log10(snr[i+1]/snr[0]), color=c)
axs[0].set_ylabel("|S/Sref| / dB")
axs[1].set_ylabel("arg S/Sref")
axs[2].set_ylabel("|Soff| / dB")
axs[3].set_ylabel("SNR(on/off) / dB")
axs[-1].set_xlabel('f / GHz')
axs[1].legend(fontsize=6)
# axs[2].legend(fontsize=8)
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
title = (
    f"signal LO={config.vnaLO/1e9:.5f}GHz   Navg {Navg}"
    f"   electric delay {config.VNA_PHASE_CORR:.3e}rad/Hz"
    f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.vna_output_gain:+.1f}dB"
    f",   {config.input_gain:+.1f}dB input gain"
    f"\nIflux {Iflux*1e3:.5}mA  Pump {pumpmeta['f']/1e9:.5f}GHz")
fig.suptitle(title, fontsize=10)
fig.savefig(fpath+'.png')
