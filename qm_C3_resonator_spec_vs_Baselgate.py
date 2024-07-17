
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

import configuration_novna as config
import qminit

qmm = qminit.connect()

#%%

from instruments.basel import BaselDACChannel

gate = BaselDACChannel(7)

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 2e-6
GATERAMP_STEPTIME = 0.02

#%%

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_resonator_vs_gate'
fpath = data_path(filename, datesuffix='_qm')

Vgate = np.concatenate([np.linspace(-3.9, -4.2, int(6e3)+1)])
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

# Vhyst = -5.24
# print(f"Gate hysteresis sweep ({abs(gate.get_voltage()-Vhyst)/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
# gate.ramp_voltage(Vhyst, 2*GATERAMP_STEP, GATERAMP_STEPTIME)

Navg = 50
f_min = 199e6
f_max = 212e6
df = 0.05e6
freqs = np.arange(f_min, f_max + df/2, df)  # + df/2 to add f_max to freqs
Nf = len(freqs)

dataS21 = np.full((Ngate, Nf), np.nan+0j)
tracetime = np.full(Ngate, np.nan)

with qua.program() as resonator_spec:
    ngate = qua.declare(int)
    n = qua.declare(int)
    f = qua.declare(int)
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    n_st = qua.declare_stream()
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()

    with qua.for_(ngate, 0, ngate < Ngate, ngate + 1):
        qua.pause()
        with qua.for_(n, 0, n < Navg, n + 1):
            with qua.for_(f, f_min, f <= f_max, f + df):  # integer loop
                qua.update_frequency('resonator', f)
                qua.wait(config.cooldown_clk, 'resonator')
                qua.measure('readout', 'resonator', None,
                        qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)
            qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(Navg, len(freqs)).map(qua.FUNCTIONS.average(0)).save_all('I')
        Q_st.buffer(Navg, len(freqs)).map(qua.FUNCTIONS.average(0)).save_all('Q')

print(f"Setting gate ({abs(gate.get_voltage()-Vgate[0])/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
gate.ramp_voltage(Vgate[0], GATERAMP_STEP, GATERAMP_STEPTIME)
print("Wait for gate to settle")
time.sleep(5)

fig, ax = plt.subplots()
img = plt2dimg(ax, Vgate, freqs, np.abs(dataS21))
ax.set_xlabel("Vgate / V")
ax.set_ylabel("resonator IF")
#ax.yaxis.set_major_formatter(EngFormatter(sep='', unit='Hz'))
fig.colorbar(img, ax=ax).set_label("|S| / linear")
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    f"resonatorLO {(config.resonatorLO)/1e9:.5f}GHz"
    f"   Navg {Navg}"
    f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
    f",   {config.resonator_input_gain:+.1f}dB input gain")
fig.suptitle(title, fontsize=8)
fig.tight_layout()


QMSLEEP = 0.05
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
job = qm.execute(resonator_spec)
res_handles = job.result_handles
Ihandle = res_handles.get('I')
Qhandle = res_handles.get('Q')
while not job.is_paused():
    mpl_pause(QMSLEEP)


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
        while not job.is_paused() and not job.status == 'completed':
            mpl_pause(QMSLEEP)
        tqm.append(time.time()-tstart)

        if i%100 == 0 or i == Ngate-1:
            I = Ihandle.fetch_all()['value']
            Q = Qhandle.fetch_all()['value']
            l = min(I.shape[0], Q.shape[0])
            dataS21[:l] = I[:l] + 1j * Q[:l]
            plt2dimg_update(img, np.abs(dataS21))

        estimator.step(i)
finally:
    job.halt()
    estimator.end()
    try: # in case of interrupt
        I = Ihandle.fetch_all()['value']
        Q = Qhandle.fetch_all()['value']
        l = min(I.shape[0], Q.shape[0])
        dataS21[:l] = I[:l] + 1j * Q[:l]
    except:
        pass
    np.savez_compressed(
        fpath, Navg=Navg, f=freqs, dataS21=dataS21, Vgate=Vgate,
        config=config.meta)
    print("Time per trace:", (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime)), 's')
    print("Time for gate set:", np.mean(tgate), "s")
    print("Time for QM execution:", np.mean(tqm), "s")

    plt2dimg_update(img, np.abs(dataS21))
    fig.tight_layout()
    # fig.savefig(fpath+'.png', dpi=300)

    Zcorr = dataS21 * np.exp(1j * freqs * config.PHASE_CORR) / config.readout_len * 2**12

    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, layout='constrained')
    img = plt2dimg(axs[0], Vgate, freqs/1e6, 10*np.log10(np.abs(dataS21)**2 * 10))
    fig.colorbar(img, ax=axs[0]).set_label("|S| / dBm")
    img = plt2dimg(axs[1], Vgate, freqs/1e6, np.angle(Zcorr), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(img, ax=axs[1]).set_label("arg S")
    axs[0].set_ylabel('resonator IF / MHz', fontsize=6)
    axs[1].set_ylabel('resonator IF / MHz', fontsize=6)
    axs[1].set_xlabel("Vgate / V")
    fig.suptitle(title, fontsize=8)
    fig.savefig(fpath+'.png', dpi=300)

#%%

dat = np.load('2024-02-20_qm/2024-02-20_17-41-16_qm_resonator_vs_gate.npz')
dat.files

fidx = np.argmin(np.abs(dat['f'] - 206.5e6))

plt.figure()
plt.plot(dat['Vgate'], np.angle(dat['dataS21'])[:,fidx])

#%%

#Shuttle
Vtarget = -4.042

step = 2e-6
steptime = 0.01
print("Ramp time:", np.abs(Vtarget - gate.get_voltage()) / step * steptime / 60, "min")
gate.ramp_voltage(Vtarget, step, steptime)

