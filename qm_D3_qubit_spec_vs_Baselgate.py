
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

from helpers import data_path, mpl_pause, DurationEstimator, plt2dimg, plt2dimg_update

import configuration as config
import qminit

qmm = qminit.connect()

#%%

from instruments.basel import BaselDACChannel

gate = BaselDACChannel(7)

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 5e-6
GATERAMP_STEPTIME = 0.02

#%%

importlib.reload(config)
importlib.reload(qminit)

qm = qmm.open_qm(config.qmconfig)
qm.octave.calibrate_element('qubit', [(config.qubitLO, config.qubitIF)])

filename = '{datetime}_qm_qubit_spec_vs_Vgate'
fpath = data_path(filename, datesuffix='_qm')

#Vgate = np.concatenate([np.linspace(-4.92, -4.93, 401)])
# Vgate = np.concatenate([np.linspace(-4.412, -4.409, 101)])
Vgate = np.concatenate([np.linspace(-5.1, -5.4, 1001)])
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

Navg = 1000

f_min = -450e6
f_max = 450e6
df = 1e6
freqs = np.arange(f_min, f_max + df/2, df)  # + df/2 to add f_max to freqs
Nf = len(freqs)

print("Expected run time per gate point:", (config.cooldown_clk*4 + config.saturation_len)*1e-9 * Nf*Navg, "s")

# Compile program with config parameters
# Align short readout pulse in middle of saturation pulse
assert config.saturation_len > config.short_readout_len
readoutwait = int(((config.saturation_len - config.short_readout_len) / 2) / 4) # cycles
print("Readoutwait", readoutwait*4, "ns")

with qua.program() as qubit_spec:
    ngate = qua.declare(int)  # variable for gate loop
    n = qua.declare(int)  # variable for average loop
    f = qua.declare(int)  # variable to sweep freqs
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    n_st = qua.declare_stream()  # stream for 'n'
    I_st = qua.declare_stream()  # stream for I
    Q_st = qua.declare_stream()  # stream for Q

    with qua.for_(ngate, 0, ngate < Ngate, ngate + 1):
        qua.pause()
        with qua.for_(n, 0, n < Navg, n + 1):
            with qua.for_(f, f_min, f <= f_max, f + df):  # integer loop
                qua.update_frequency('qubit', f)
                qua.reset_phase('resonator')
                qua.wait(config.cooldown_clk, 'resonator')
                qua.align()
                qua.play('saturation', 'qubit')
                qua.wait(readoutwait, 'resonator')
                qua.measure('short_readout', 'resonator', None,
                        qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)
            qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        # I_st.buffer(Navg, len(freqs)).average(0).save_all('I')
        # Q_st.buffer(Navg, len(freqs)).average(0).save_all('Q')
        I_st.buffer(Navg, len(freqs)).map(qua.FUNCTIONS.average(0)).save_all('I')
        Q_st.buffer(Navg, len(freqs)).map(qua.FUNCTIONS.average(0)).save_all('Q')


# Set gate
print(f"Setting gate ({abs(gate.get_voltage()-Vgate[0])/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
gate.ramp_voltage(Vgate[0], GATERAMP_STEP, GATERAMP_STEPTIME)
print("Wait for gate to settle")
time.sleep(10)

# Prepare live plot
dataS21 = np.full((Ngate, Nf), np.nan+0j)
tracetime = np.full(Ngate, np.nan)
f2 = freqs
fig, ax = plt.subplots()
img = plt2dimg(ax, Vgate, f2, np.angle(dataS21))
ax.set_xlabel("Vgate / V")
ax.set_ylabel("f2")
#ax.yaxis.set_major_formatter(EngFormatter(sep='', unit='Hz'))
fig.colorbar(img, ax=ax).set_label("arg(S21)")
readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    f"resonator {(config.resonatorLO+config.resonatorIF)/1e9:f}GHz"
    f"   qubit LO {config.qubitLO/1e9:.3f}GHz"
    f"   Navg {Navg}"
    f"\n{config.short_readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain+config.short_readout_amp_gain:+.1f}dB"
    f"\n{config.saturation_len}ns saturation pulse at {saturationpower:.1f}dBm{config.qubit_output_gain:+.1f}dB")
fig.suptitle(title, fontsize=8)
fig.tight_layout()

# Start QM job
QMSLEEP = 0.01 # seconds
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config, short_readout_gain=True)
qminit.octave_setup_qubit(qm, config)
job = qm.execute(qubit_spec)
res_handles = job.result_handles
#nhandle = res_handles.get('iteration')
Ihandle = res_handles.get('I')
Qhandle = res_handles.get('Q')
while not job.is_paused(): # Wait for first pause()
    time.sleep(QMSLEEP)

tgate = []
tqm = []
estimator = DurationEstimator(Ngate)
try:
    for i in range(Ngate):
        tstart = time.time()
        gate.ramp_voltage(Vgate[i], GATERAMP_STEP, GATERAMP_STEPTIME)
        tgate.append(time.time()-tstart)
        tracetime[i] = time.time()

        tstart = time.time()
        job.resume()
        # (after last resume the job doesn't pause but exit)
        while not job.is_paused() and not job.status == 'completed':
            mpl_pause(QMSLEEP)
        tqm.append(time.time()-tstart)
        if i % 1 == 0:
            #n = nhandle.fetch_all() + 1
            I = Ihandle.fetch_all()['value']
            Q = Qhandle.fetch_all()['value']
            l = min(I.shape[0], Q.shape[0])
            dataS21[:l] = I[:l] + 1j * Q[:l]
            estimator.step(i)
            plt2dimg_update(img, np.unwrap(np.unwrap(np.angle(dataS21), axis=1), axis=0))
except Exception as e:
    job.halt()
    raise e
finally:
    estimator.end()

    if i == Ngate-1:
        res_handles.wait_for_all_values()
    I = Ihandle.fetch_all()['value']
    Q = Qhandle.fetch_all()['value']
    assert I.shape == Q.shape
    dataS21[:I.shape[0]] = I + 1j * Q

    np.savez_compressed(
        fpath, Navg=Navg, f=freqs, dataS21=dataS21,
        config=config.meta)
    ttime = (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime))
    print("Time per trace:", ttime, 's')
    print("Time for gate set:", np.mean(tgate), "s")
    print("Time for QM execution:", np.mean(tqm), "s")

    arg = np.unwrap(np.unwrap(np.angle(dataS21), axis=1), axis=0)
    plt2dimg_update(img, arg)
    fig.tight_layout()
    fig.savefig(fpath+'.png', dpi=300)

    flatarg = arg - np.mean(arg, axis=1)[:, None]

    fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1,3]}, layout='constrained')
    axs[0].plot(Vgate, np.mean(arg, axis=1))
    axs[0].set_ylabel('<arg S> at VNA center', fontsize=6)
    img = plt2dimg(axs[1], Vgate, f2 / 1e6, flatarg)
    axs[1].set_xlabel("Vgate / V")
    axs[1].set_ylabel("f2 / MHz")
    fig.colorbar(img, ax=axs[1]).set_label("arg(S21) - avg")
    fig.suptitle(title, fontsize=8)
    fig.savefig(fpath+'_flat.png', dpi=300)

#%%

Zcorr = dataS21 / config.short_readout_len * 2**12

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(freqs/1e6, 10*np.log10((np.abs(Zcorr))**2 * 10))
ax2.plot(freqs/1e6, np.unwrap(np.angle(Zcorr)))
ax1.grid(), ax2.grid()
ax1.set_ylabel('|S| / dB')
ax2.set_xlabel('Qubit IF / MHz')
ax2.set_ylabel('Phase / rad')
readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
fig.suptitle(
    f"resonator {(config.resonatorLO+config.resonatorIF)/1e9:f}GHz"
    f"   qubit LO {config.qubitLO/1e9:.3f}GHz"
    f"   Navg {Navg}"
    f"\n{config.short_readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain+config.short_readout_amp_gain:+.1f}dB"
    f"\n{config.saturation_len}ns saturation pulse at {saturationpower:.1f}dBm{config.qubit_output_gain:+.1f}dB",
    fontsize=8)
fig.tight_layout()
fig.savefig(fpath+'.png')

#%%

#Shuttle
Vtarget = -5.2745     # 6.6173

step = 30e-6
steptime = 0.02
print("Ramp time:", np.abs(Vtarget - gate.get_voltage()) / step * steptime / 60, "min")
gate.ramp_voltage(Vtarget, step, steptime)
