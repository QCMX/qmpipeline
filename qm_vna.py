
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

from helpers import data_path, mpl_pause

import configuration as config
import qminit

qmm = qminit.connect()

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

filename = '{datetime}_qm_vna_spec'
fpath = data_path(filename, datesuffix='_qm')

try:
    Vgate = gate.get_voltage()
except:
    Vgate = np.nan

Navg = 100

f_min = -451e6 # 102e6
f_max = +451e6 # 112e6
df = 5e6
freqs = np.arange(f_min, f_max + df/2, df)  # + df/2 to add f_max to freqs

with qua.program() as vna:
    n = qua.declare(int)  # variable for average loop
    n_st = qua.declare_stream()  # stream for 'n'
    f = qua.declare(int)  # variable to sweep freqs
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    I_st = qua.declare_stream()  # stream for I
    Q_st = qua.declare_stream()  # stream for Q
    rand = qua.lib.Random()

    with qua.for_(n, 0, n < Navg, n + 1):
        with qua.for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            qua.update_frequency('vna', f)  # update frequency of vna element
            qua.reset_phase('vna')
            qua.wait(config.cooldown_clk, 'vna')  # wait for resonator to decay
            qua.wait(rand.rand_int(50)+4, 'vna')
            qua.align()
            #play('saturation', 'qubit')
            # qua.align()
            qua.measure('readout', 'vna', None,
                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(freqs)).average().save('I')
        Q_st.buffer(len(freqs)).average().save('Q')


qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_vna(qm, config)
job = qm.execute(vna)

res_handles = job.result_handles
I_handle = res_handles.get('I')
Q_handle = res_handles.get('Q')
iteration_handle = res_handles.get('iteration')
I_handle.wait_for_values(1), Q_handle.wait_for_values(1)
iteration_handle.wait_for_values(1)

# Live plotting
fig, ax = plt.subplots()
line, = ax.plot(freqs/1e6, np.full(len(freqs), np.nan), label="|Z|")
ax.set_title('spectroscopy analysis')
ax.set_xlabel('IF [MHz]')
ax.set_ylabel('demod signal [a.u.]')
fig.show()
try:
    while res_handles.is_processing():
        iteration = iteration_handle.fetch_all() + 1
        I, Q = I_handle.fetch_all(), Q_handle.fetch_all()
        Z = I + Q*1j
        line.set_ydata(np.abs(Z))
        ax.relim(), ax.autoscale(), ax.autoscale_view()
        print(iteration)
        mpl_pause(0.5)
except KeyboardInterrupt:
    job.halt()

# Get and save data
Nactual = iteration_handle.fetch_all()+1
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
Zraw = I + 1j * Q

np.savez_compressed(
    fpath, Navg=Navg, Nactual=Nactual, freqs=freqs, Zraw=Zraw,
    config=config.meta)

# Final plot
line.set_ydata(np.abs(Zraw))
ax.plot(freqs/1e6, I, label='I')
ax.plot(freqs/1e6, Q, label='Q')
ax.legend(), ax.grid(), fig.tight_layout()

print(f"Raw amplitude: {np.mean(np.abs(Zraw))}")
print(f"Raw stdev: I {np.std(I)}   Q {np.std(Q)}")
print(f"Raw stdev (single shot): I {np.std(I)*np.sqrt(Navg)}   Q {np.std(Q)*np.sqrt(Navg)}")

phasediff = np.diff(np.unwrap(np.angle(Zraw))[[0,-1]])[0]
print(f"Raw end-to-end phase diff: {phasediff} / rad")
#print(f"To correct use: {-phasediff / (freqs[-1]-freqs[0])} rad/Hz (including no shift)")
print(f"To correct use: {(-phasediff) / (freqs[-1]-freqs[0])} rad/Hz")

Zcorr = Zraw * np.exp(1j * freqs * config.VNA_PHASE_CORR) / config.readout_len * 2**12

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, gridspec_kw={'height_ratios': [3,3,1]})
ax1.plot(freqs/1e6, 10*np.log10((np.abs(Zcorr))**2 * 10))
ax2.plot(freqs/1e6, np.unwrap(np.angle(Zcorr)))
ax1.grid(), ax2.grid()
ax1.set_ylabel(f"|S| / dBm", fontsize=8)
ax2.set_ylabel('arg S/ rad', fontsize=8)
ax2.set_xlabel('VNA IF / MHz', fontsize=8)
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
fig.suptitle(
    f"LO={config.vnaLO/1e9:.5f}GHz"
    f"   Navg {Nactual}"
    #f"   electric delay {config.PHASE_CORR/np.pi*180*1e3:.2e}deg/kHz",
    f"   electric delay {config.PHASE_CORR:.3e}rad/Hz"
    f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.vna_output_gain:+.1f}dB"
    f",   {config.input_gain:+.1f}dB input gain"
    f"\nVgate={Vgate:.6f}V",
    fontsize=10)
fig.tight_layout()
fig.savefig(fpath+'.png')

#%% Resonator Sectoscopy

element = 'qubit'
qm.octave.set_lo_frequency(element, config.qubitLO)
qm.octave.set_lo_source(element, octave.OctaveLOSource.Internal)
qm.octave.set_rf_output_gain(element, config.qubit_output_gain)
qm.octave.set_rf_output_mode(element, octave.RFOutputMode.on)

# Align readout pulse in middle of saturation pulse
assert config.saturation_len > config.readout_len
readoutwait = int(((config.saturation_len - config.readout_len) / 2) / 4) # cycles
print("Readoutwait", readoutwait*4, "ns")

with qua.program() as resonator_spec_saturation:
    n = qua.declare(int)  # variable for average loop
    n_st = qua.declare_stream()  # stream for 'n'
    f = qua.declare(int)  # variable to sweep freqs
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    I_st = qua.declare_stream()  # stream for I
    Q_st = qua.declare_stream()  # stream for Q

    with qua.for_(n, 0, n < Navg, n + 1):
        with qua.for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            qua.update_frequency('resonator', f)  # update frequency of resonator element
            qua.reset_phase('resonator')
            qua.wait(config.cooldown_clk, 'resonator') # wait for resonator to decay
            qua.align()
            qua.play('saturation', 'qubit')
            qua.wait(readoutwait)
            qua.measure('readout', 'resonator', None,
                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(freqs)).average().save('I')
        Q_st.buffer(len(freqs)).average().save('Q')


# Execute program
job = qm.execute(resonator_spec_saturation)

res_handles = job.result_handles  # get access to handles
I_handle = res_handles.get('I')
I_handle.wait_for_values(1)
Q_handle = res_handles.get('Q')
Q_handle.wait_for_values(1)
iteration_handle = res_handles.get('iteration')
iteration_handle.wait_for_values(1)

# Live plotting
plt.figure()
while res_handles.is_processing():
    try:
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        iteration = iteration_handle.fetch_all() + 1
        plt.title('resonator spectroscopy analysis')
        Z = I + Q*1j
        plt.clf()
        plt.plot(freqs, np.sqrt(np.abs(Zrawnonsat)), '--')
        plt.plot(freqs, np.sqrt(np.abs(Z)))
        plt.xlabel('IF [Hz]')
        plt.ylabel('demod signal [a.u.]')
        plt.pause(0.5)
        print(iteration)
    except Exception:
        pass

# Get and save data
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
Zraw = I + 1j * Q

np.savez_compressed(
    fpath+'_saturation',
    Navg=Navg, freqs=freqs, Zraw=Zraw,
    Zrawnosaturation=Zrawnonsat,
    config=config.meta)

# Final plot
plt.clf()
plt.title('resonator spectroscopy analysis')
plt.plot(freqs, np.abs(Zrawnonsat), '--C0', label='|Z| no saturation')
plt.plot(freqs, Inonsat, '--C1', label='I')
plt.plot(freqs, Qnonsat, '--C2', label='Q')
plt.plot(freqs, np.abs(Zraw), 'C0', label='|Z| with saturation pulse')
plt.plot(freqs, I, 'C1', label='I')
plt.plot(freqs, Q, 'C2', label='Q')
plt.xlabel('IF [Hz]')
plt.ylabel('demod signal [a.u.]')
plt.legend(), plt.grid(), plt.tight_layout()


Zcorrnonsat = Zrawnonsat * np.exp(1j * freqs * config.PHASE_CORR)
Zcorr = Zraw * np.exp(1j * freqs * config.PHASE_CORR)

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(freqs, 20*np.log10(np.abs(Zcorrnonsat)), 'C0', label='no saturation pulse')
ax1.plot(freqs, 20*np.log10(np.abs(Zcorr)), 'C1', label='saturation pulse')
ax2.plot(freqs, np.unwrap(np.angle(Zcorrnonsat)), 'C0')
ax2.plot(freqs, np.unwrap(np.angle(Zcorr)), 'C1')
ax1.grid(), ax2.grid()
ax1.legend(fontsize=8)
ax1.set_ylabel('|S| / dB')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Phase / rad')
fig.suptitle(
    f"LO={config.resonatorLO/1e9:.3f}GHz"
    f"   Navg {Navg}"
    #f"   electric delay {config.PHASE_CORR/np.pi*180*1e3:.2e}deg/kHz",
    f"   electric delay {config.PHASE_CORR:.3e}rad/Hz",
    fontsize=10)
fig.tight_layout()
fig.savefig(fpath+'_saturation.png')
