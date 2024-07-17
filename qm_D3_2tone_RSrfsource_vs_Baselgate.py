
# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# - electrical delay correction (config.PHASE_CORR)
# - resonator IF

import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import qm.qua as qua
import qm.octave as octave
from qualang_tools.loops import from_array

from helpers import data_path, mpl_pause, DurationEstimator, plt2dimg, plt2dimg_update

import configuration_novna as config
import qminit

qmm = qminit.connect()

#%%

from instruments.basel import BaselDACChannel

gate = BaselDACChannel(7)

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 2e-6
GATERAMP_STEPTIME = 0.01

#%%

from RsInstrument import RsInstrument

rfsource = RsInstrument('TCPIP::169.254.2.30::INSTR', id_query=True, reset=False)
rfsource.instrument_status_checking = False # faster
rfsource.opc_query_after_write = False # faster
rfsource.write_str_with_opc(":output off")

#%%

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_2tone_vs_Vgate'
fpath = data_path(filename, datesuffix='_qm')

Vgate = np.linspace(-3.92, -4.1, 901)
Vgate = np.linspace(-3.96, -3.98, 101)
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

# QM resume seems to have about 20ms overhead
f = np.arange(200e6, 212e6, 0.2e6)
Navg = 100
GATE_SETTLING_TIME = 2 # s

# RF source freq set has 6.5ms overhead
f2 = np.arange(0.1e9, 4.3e9, 10e6)
power2 = np.array([0, 10])

#print("Expected run time per gate point:", (config.cooldown_clk*4 + config.saturation_len)*1e-9 * Nf*Navg, "s")

with qua.program() as prog2tone:
    ngate = qua.declare(int)
    npower2 = qua.declare(int)
    nf2 = qua.declare(int)
    n = qua.declare(int)
    fr = qua.declare(int)
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    with qua.for_(ngate, 0, ngate < Ngate, ngate + 1):
        with qua.for_(npower2, 0, npower2 < power2.size, npower2 + 1):
            with qua.for_(nf2, 0, nf2 < f2.size, nf2 + 1):
                qua.pause()
                with qua.for_(n, 0, n < Navg, n + 1):
                    with qua.for_(*from_array(fr, f)):
                        qua.update_frequency('resonator', fr)
                        qua.wait(config.cooldown_clk, 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator')
                        qua.measure('readout', 'resonator', None,
                                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)

        # Reference, 2nd tone OFF
        qua.pause()
        with qua.for_(n, 0, n < Navg, n + 1):
            with qua.for_(*from_array(fr, f)):
                qua.update_frequency('resonator', fr)
                qua.wait(config.cooldown_clk, 'resonator')
                qua.wait(rand.rand_int(50)+4, 'resonator')
                qua.measure('readout', 'resonator', None,
                        qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)

    with qua.stream_processing():
        I_st.buffer(power2.size*f2.size+1, Navg, f.size).map(qua.FUNCTIONS.average(1)).save_all('I')
        Q_st.buffer(power2.size*f2.size+1, Navg, f.size).map(qua.FUNCTIONS.average(1)).save_all('Q')


# Set gate
print(f"Setting gate ({abs(gate.get_voltage()-Vgate[0])/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
gate.ramp_voltage(Vgate[0], GATERAMP_STEP, GATERAMP_STEPTIME)
print("Wait for gate to settle")
time.sleep(5)

S21ref = np.full((Ngate, f.size), np.nan+0j)
S21 = np.full((Ngate, power2.size, f2.size, f.size), np.nan+0j)
tracetime = np.full(Ngate, np.nan)

# Live plot
fig, axs = plt.subplots(nrows=power2.size+1, sharex=True, layout='constrained')
cavityimg = plt2dimg(axs[0], Vgate, f/1e6, np.full((Vgate.size, f.size), np.nan))
fig.colorbar(cavityimg, ax=axs[0])
axs[0].set_ylabel("cavity IF / MHz")
pimgs = []
for j in range(power2.size):
    img = plt2dimg(axs[j+1], Vgate, f2/1e9, np.full((Vgate.size, f2.size), np.nan))
    fig.colorbar(img, ax=axs[j+1])
    axs[j+1].set_title(f"P2={power2[j]:+.1f}dBm")
    pimgs.append(img)
    if j > 0:
        axs[j+1].sharey(axs[1])
    axs[j+1].set_ylabel("f2 / GHz")
axs[-1].set_xlabel("Vgate / V")
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
title = (
    f"resonator {(config.resonatorLO+config.resonatorIF)/1e9:f}GHz   Navg {Navg}"
    f"\n{config.readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB")
fig.suptitle(title, fontsize=8)

# Setup drive
rfsource.write_str_with_opc(f':source:power {power2[0]:f}')
rfsource.write_str_with_opc(f':source:freq {f2[0]:f}')
rfsource.write_str_with_opc(":output on")

# Calibrate output mixers
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
cal = qm.octave.calibrate_element('resonator', [(config.resonatorLO, config.resonatorIF)])

# Start QM job
QMSLEEP = 0.01 # seconds
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
job = qm.execute(prog2tone)
Ihandle = job.result_handles.get('I')
Qhandle = job.result_handles.get('Q')
while not job.is_paused(): # Wait for first pause()
    time.sleep(QMSLEEP)

tgate = []
trf = []
tqm = []
estimator = DurationEstimator(Ngate)
try:
    for i in range(Ngate):
        print("Gate", Vgate[i], "V")
        tstart = time.time()
        gate.ramp_voltage(Vgate[i], GATERAMP_STEP, GATERAMP_STEPTIME)
        time.sleep(GATE_SETTLING_TIME) # settle
        tgate.append(time.time()-tstart)
        tracetime[i] = time.time()

        rfsource.write_str_with_opc(f':source:power {power2[0]:f}')
        rfsource.write_str_with_opc(":output on")

        for j in range(power2.size):
            print("   ", power2[j], "dBm")
            rfsource.write_str_with_opc(f':source:power {power2[j]:f}')
            for k in range(f2.size):
                tstart = time.time()
                rfsource.write(f":source:freq {f2[k]}")
                trf.append(time.time()-tstart)

                tstart = time.time()
                job.resume()
                # (after last resume the job doesn't pause but exit)
                while not job.is_paused() and not job.status == 'completed':
                    mpl_pause(QMSLEEP)
                tqm.append(time.time()-tstart)

        # Reference
        rfsource.write_str_with_opc(":output off")
        tstart = time.time()
        job.resume()
        # (after last resume the job doesn't pause but exit)
        while not job.is_paused() and not job.status == 'completed':
            mpl_pause(QMSLEEP)
        tqm.append(time.time()-tstart)

        while Ihandle.count_so_far() <= i or Qhandle.count_so_far() <= i:
            mpl_pause(QMSLEEP)
        Z = (Ihandle.fetch(i)['value'][0] + 1j * Qhandle.fetch(i)['value'][0]) / config.readout_len * 2**12
        S21ref[i] = Z[-1]
        S21[i] = Z[:-1].reshape(power2.size, f2.size, f.size)

        # Signal is distance from reference
        signal = np.sum(np.abs(S21 - S21ref[:,None,None,:]), axis=-1)
        plt2dimg_update(cavityimg, np.abs(S21ref))
        for j in range(power2.size):
            plt2dimg_update(pimgs[j], signal[:,j,:])
        #     estimator.step(i)
except Exception as e:
    job.halt()
    raise e
finally:
    estimator.end()

    np.savez_compressed(
        fpath, Navg=Navg, f=f, f2=f2, power2=power2, Vgate=Vgate, S21=S21,
        config=config.meta)

    ttime = (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime))
    print("Time per gate point:", ttime, 's')
    print("Time for gate set:", np.mean(tgate), "s")
    print("Time for RF set:", np.mean(trf), "s")
    print("Time for QM execution:", np.mean(tqm), "s")
    
    fig.savefig(fpath+'.png', dpi=300)

#%%

#Shuttle
Vtarget = -3.975     # 6.6173

step = 1e-6
steptime = 0.02
print("Ramp time:", np.abs(Vtarget - gate.get_voltage()) / step * steptime / 60, "min")
gate.ramp_voltage(Vtarget, step, steptime)
