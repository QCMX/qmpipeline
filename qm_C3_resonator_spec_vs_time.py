
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

import importlib
import BlueforsPoll
importlib.reload(BlueforsPoll)

from BlueforsPoll import BlueforsThermoPoll
from instruments.blueforsthermometer import BlueforsOldThermometer

try:
    poll.stop()
except: pass

thermo = BlueforsOldThermometer()
poll = BlueforsThermoPoll.make_poll(thermo, ['MXC', '4K'], interval=5)

#%%
from datetime import datetime

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_resonator_vs_time'
fpath = data_path(filename, datesuffix='_qm')

Nt = 30000
dt = 2 # sec
print(Nt * dt / 3600, "hours")

try:
    Vgate = gate.get_voltage()
except:
    Vgate = np.nan

Navg = 200
f_min = 200e6
f_max = 209e6
df = 0.05e6
freqs = np.arange(f_min, f_max + df/2, df)  # + df/2 to add f_max to freqs
Nf = len(freqs)

dataS21 = np.full((Nt, Nf), np.nan+0j)
tracetime = np.full(Nt, np.nan)
tracetemp = np.full(Nt, np.nan)

with qua.program() as resonator_spec:
    ngate = qua.declare(int)
    n = qua.declare(int)
    f = qua.declare(int)
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    n_st = qua.declare_stream()
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()

    with qua.for_(ngate, 0, ngate < Nt, ngate + 1):
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


fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, layout='constrained')
ts = np.arange(Nt)*dt / 3600
img = plt2dimg(ax, ts, freqs, np.abs(dataS21))
ax.set_xlabel("Time / hours")
ax.set_ylabel("resonator IF")
#ax.yaxis.set_major_formatter(EngFormatter(sep='', unit='Hz'))
fig.colorbar(img, ax=ax).set_label("|S| / linear")
tline, = ax2.plot(ts, tracetemp)
ax2.set_ylabel('TMXC / K')

readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    f"resonatorLO {(config.resonatorLO)/1e9:.5f}GHz"
    f"   Navg {Navg}"
    f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
    f",   {config.resonator_input_gain:+.1f}dB input gain")
fig.suptitle(title, fontsize=8)


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
estimator = DurationEstimator(Nt)
try:
    for i in range(Nt):
        tracetime[i] = time.time()

        tstart = time.time()
        job.resume()
        while not job.is_paused() and not job.status == 'completed':
            mpl_pause(QMSLEEP)
        tqm.append(time.time()-tstart)
        tracetemp[i] = poll.get_temp('MXC')[0]

        if i%100 == 0 or i == Nt-1:
            I = Ihandle.fetch_all()['value']
            Q = Qhandle.fetch_all()['value']
            l = min(I.shape[0], Q.shape[0])
            dataS21[:l] = I[:l] + 1j * Q[:l]
            plt2dimg_update(img, np.abs(dataS21))
            tline.set_ydata(tracetemp)
            ax2.relim(), ax2.autoscale(), ax2.autoscale_view()

        estimator.step(i)
        mpl_pause(max(QMSLEEP, dt - (time.time() - tstart)))
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
        fpath, Navg=Navg, f=freqs, dataS21=dataS21, Vgate=Vgate, Nt=Nt,
        config=config.meta)
    print("Time per trace:", (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime)), 's')
    print("Time for gate set:", np.mean(tgate), "s")
    print("Time for QM execution:", np.mean(tqm), "s")

    plt2dimg_update(img, np.abs(dataS21))
    tline.set_ydata(tracetemp)
    fig.savefig(fpath+'.png', dpi=300)

#%%

dat = np.load('2024-02-20_qm/2024-02-20_17-41-16_qm_resonator_vs_gate.npz')
dat.files

fidx = np.argmin(np.abs(dat['f'] - 206.5e6))

plt.figure()
plt.plot(dat['Vgate'], np.angle(dat['dataS21'])[:,fidx])

#%%

from instruments.basel import BaselDACChannel

gate = BaselDACChannel(7)

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 2e-6
GATERAMP_STEPTIME = 0.02

#%%
#Shuttle
Vtarget = -5.281 # 6.6173

step = 2e-6
steptime = 0.02
print("Ramp time:", np.abs(Vtarget - gate.get_voltage()) / step * steptime / 60, "min")
gate.ramp_voltage(Vtarget, step, steptime)

