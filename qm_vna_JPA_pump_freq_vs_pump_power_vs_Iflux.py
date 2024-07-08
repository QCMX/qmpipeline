"""
Script to tune JPA.

Important parameters are
- signal frequency (usually known)
- signal power (limited by amplifier compression, usually known)
- JPA pump frequency (around twice signal frequency +/- bandwidth)
- JPA pump power
- JPA flux bias current

The pump frequency and power are the most sensitive parameters, then the flux
bias current. For this reason this script scans pump parameters finely and
repeats measurement at multiple but not as many flux bias points.

Live plot shows amplifier gain.
Second plot after measurement shows SNR.

SNR is avg(|signal|) / std(|signal|). Note that this is roughly the inverse
of std(arg(signal)), since std(arg(signal)) = tan(noise / signal) for small
noise compared to signal.

The SNR is for a single readout measurement, since we take the variance between
demodulation results. The signal values are the averaged ones. So the SNR
does not include the SNR improvement due to averaging here.

"""
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import qm.qua as qua
import qm.octave as octave
# from qualang_tools.loops import from_array

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

print("Running calibration on vna output...")
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

print("Running calibration on resonator output...")
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
cal = qm.octave.calibrate_element('resonator', [(config.resonatorLO, config.resonatorIF)])

#%%

importlib.reload(config)
importlib.reload(qminit)

element = 'resonator'
filename = '{datetime}_qm_JPA_pump_power_vs_pump_freq_vs_Iflux'
fpath = data_path(filename, datesuffix='_qm')

#Iflux = np.array([0e-3, 0.01e-3, 0.02e-3, 0.025e-3, 0.027e-3, 0.03e-3, 0.032e-3]) # A
#Iflux = np.array([0.023e-3, 0.024e-3, 0.025e-3, 0.026e-3, 0.027e-3, 0.028e-3, 0.029e-3, 0.03e-3]) # A
Iflux = np.array([0.025e-3, 0.026e-3]) # A
if element == 'vna':
    fsignal = config.vnaLO + config.vnaIF
else:
    fsignal = config.resonatorLO + config.resonatorIF

Navg = 100
fpump = np.arange(9.5e9, 11.5e9, 5e6)
fpump = np.arange(10.3e9, 10.7e9, 1e6)
fpump = np.arange(10.44e9, 10.54e9, 0.4e6)
Ppump = np.arange(-17, -10, 0.2)

# Ramp to flux
assert np.all(np.abs(Iflux[0]) < 1e-3) # Limit 1mA thermocoax
if fluxbias.current() > Iflux[0]:
    print("Hysteresis, start from Iflux=0mA")
    fluxbias.ramp_current(0, FLUXRAMP_STEP, FLUXRAMP_STEPTIME)
print(f"Setting flux current ({abs(fluxbias.current()-Iflux[0])/FLUXRAMP_STEP*FLUXRAMP_STEPTIME/60:.1f}min)")
fluxbias.ramp_current(Iflux[0], FLUXRAMP_STEP, FLUXRAMP_STEPTIME)
time.sleep(2) # settle

# Define program using settings above
with qua.program() as vnaprog:
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
        qua.wait(rand.rand_int(50)+4, element)
        qua.measure('readout', element, None,
                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
        qua.save(I, I_st)
        qua.save(Q, Q_st)

    with qua.for_(nPpump, 0, nPpump < Ppump.size, nPpump + 1):
        with qua.for_(nfpump, 0, nfpump < fpump.size, nfpump + 1):
            qua.pause()
            with qua.for_(n, 0, n < Navg, n + 1):
                # qua.wait(config.cooldown_clk, 'vna') # not really necessary, VNA is CW measurement
                qua.wait(rand.rand_int(50)+4, element)
                qua.measure('readout', element, None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)

    with qua.stream_processing():
        Iavg = I_st.buffer(Navg).map(qua.FUNCTIONS.average(0))
        Qavg = Q_st.buffer(Navg).map(qua.FUNCTIONS.average(0))
        Iavg.save_all('I')
        Qavg.save_all('Q')
        # Calculate variance = <x*x> - <x> * <x>
        # We have here fixed point numbers with 4.28 bits, i.e. in range [-8, 8)
        # with precision 2**-28 = 3.2e-9
        # Fixed point numbers should not suffer from cancellation in this expression.
        ((I_st.buffer(Navg)*I_st.buffer(Navg)).map(qua.FUNCTIONS.average(0)) - Iavg*Iavg).save_all('Ivar')
        ((Q_st.buffer(Navg)*Q_st.buffer(Navg)).map(qua.FUNCTIONS.average(0)) - Qavg*Qavg).save_all('Qvar')

refsignal = np.full(Iflux.size, np.nan+0j)
refsignalvar = np.full(Iflux.size, np.nan+0j)
S21 = np.full((Iflux.size, Ppump.size, fpump.size), np.nan+0j)
S21var = np.full((Iflux.size, Ppump.size, fpump.size), np.nan+0j) # variance along I and Q independently
tracetime = np.full((Iflux.size, Ppump.size, fpump.size), np.nan)

# Live plotting
from matplotlib.colors import CenteredNorm
# from mpl_toolkits.axes_grid1 import ImageGrid
nrows = int(np.sqrt(Iflux.size))
ncols = int(np.ceil(Iflux.size/nrows))
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, layout='constrained', squeeze=False, figsize=(12,8))
# fig = plt.figure()
# imggrid = ImageGrid(fig, 111, ngrids=Iflux.size, nrows_ncols=(nrows, ncols), share_all=True, cbar_mode=None, aspect=False)
imgs = [None] * Iflux.size
# fig.colorbar(img, label="Gain  S/Sref  [dB]")
axs[0,0].set_ylabel('Pump power / dBm')
axs[-1,0].set_xlabel('Pump freq / GHz')
for k, I in enumerate(Iflux):
    axs[k//ncols,k%ncols].set_title(f"Iflux {I*1e3:.5f} mA", fontsize=10)
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
if element == 'vna':
    title = (
        f"signal {config.vnaLO/1e9:.5f}GHz+{config.vnaIF/1e6:.3f}MHz   Navg {Navg}"
        f"\n{config.readout_len/1e3}us readout at {readoutpower:.1f}dBm{config.vna_output_gain:+.1f}dB"
        f",   {config.input_gain:+.1f}dB input gain"
        "\nColorbar:  Gain  S/Sref  [dB]")
else:
    title = (
        f"signal {config.resonatorLO/1e9:.5f}GHz+{config.resonatorIF/1e6:.3f}MHz   Navg {Navg}"
        f"\n{config.readout_len/1e3}us readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
        f",   {config.input_gain:+.1f}dB input gain"
        "\nColorbar:  Gain  S/Sref  [dB]")
fig.suptitle(title, fontsize=10)
fig.show()


QMSLEEP = 0.05
estimator = DurationEstimator(Iflux.size*Ppump.size*fpump.size)
try:
    for k in range(Iflux.size):
        print(f"Setting flux current ({abs(fluxbias.current()-Iflux[k])/FLUXRAMP_STEP*FLUXRAMP_STEPTIME/60:.1f}min)")
        fluxbias.ramp_current(Iflux[k], FLUXRAMP_STEP, FLUXRAMP_STEPTIME)
        time.sleep(2) # settle

        # Start QM program
        qm = qmm.open_qm(config.qmconfig)
        if element == 'vna':
            qminit.octave_setup_vna(qm, config)
        else:
            qminit.octave_setup_resonator(qm, config)
        job = qm.execute(vnaprog)

        # Prepare plot
        imgs[k] = plt2dimg(
            axs[k//ncols,k%ncols], fpump/1e9, Ppump,
            20*np.log10(np.abs(S21[k].T/refsignal[k])),
            norm=CenteredNorm(), cmap='coolwarm')
        fig.colorbar(imgs[k], ax=axs[k//ncols,k%ncols], orientation='horizontal', shrink=0.8)

        Ihandle, Qhandle = job.result_handles.get('I'), job.result_handles.get('Q')
        Ivarhandle, Qvarhandle = job.result_handles.get('Ivar'), job.result_handles.get('Qvar')

        while not job.is_paused():
            mpl_pause(QMSLEEP)

        # Acquire reference, pump off
        rfsource.write_str_with_opc(":output off")
        job.resume()
        while not job.is_paused() and not job.status == 'completed':
            mpl_pause(QMSLEEP)

        # Acquire signal, pump on
        rfsource.write_str_with_opc(f':source:power {Ppump[0]:f}')
        rfsource.write_str_with_opc(f':source:freq {fpump[0]/1e9:f}ghz')
        rfsource.write_str_with_opc(":output on")
        for i in range(Ppump.size):
            rfsource.write_str_with_opc(f":source:power {Ppump[i]:f}")
            for j in range(fpump.size):
                rfsource.write_str_with_opc(f':source:freq {fpump[j]/1e9:f}ghz')

                # Acquire a trace
                tracetime[k,i,j] = time.time()
                job.resume()
                while not job.is_paused() and not job.status == 'completed':
                    mpl_pause(QMSLEEP)
    
                if ((k*Ppump.size + i)*fpump.size + j) % 20 == 0:
                    estimator.step((k*Ppump.size + i)*fpump.size + j)

            # Retrieve data and plot
            while (Ihandle.count_so_far() < (i+1)*fpump.size+1
                   or Qhandle.count_so_far() < (i+1)*fpump.size+1
                   or Ivarhandle.count_so_far() < (i+1)*fpump.size+1
                   or Qvarhandle.count_so_far() < (i+1)*fpump.size+1):
                mpl_pause(QMSLEEP)
            # signal
            I, Q = Ihandle.fetch_all()['value'], Qhandle.fetch_all()['value']
            refsignal[k] = I[0] + 1j * Q[0]
            S21[k,:i+1] = (I[1:] + 1j * Q[1:]).reshape(i+1, fpump.size)
            plt2dimg_update(imgs[k], 20*np.log10(np.abs(S21[k]/refsignal[k])).T)
            # variance: not plotted live, no need to retrieve here

        Ihandle.wait_for_all_values()
        Qhandle.wait_for_all_values()
        Ivarhandle.wait_for_all_values()
        Qvarhandle.wait_for_all_values()
        # signal
        I, Q = Ihandle.fetch_all()['value'], Qhandle.fetch_all()['value']
        refsignal[k] = I[0] + 1j * Q[0]
        S21[k,:i+1] = (I[1:] + 1j * Q[1:]).reshape(i+1, fpump.size)
        plt2dimg_update(imgs[k], 20*np.log10(np.abs(S21[k]/refsignal[k])).T)
        # variance
        Ivar, Qvar = Ivarhandle.fetch_all()['value'], Qvarhandle.fetch_all()['value']
        refsignalvar[k] = Ivar[0] + 1j*Qvar[0]
        S21var[k,:i+1] = (Ivar[1:] + 1j*Qvar[1:]).reshape(i+1, fpump.size)
finally:
    job.halt()
    estimator.end()
    try: # in case of interrupt
        Ihandle.wait_for_all_values()
        Qhandle.wait_for_all_values()
        Ivarhandle.wait_for_all_values()
        Qvarhandle.wait_for_all_values()
        # signal
        I, Q = Ihandle.fetch_all()['value'], Qhandle.fetch_all()['value']
        refsignal[k] = I[0] + 1j * Q[0]
        S21[k] = np.pad(I[1:]+1j*Q[1:], pad_width=(0,Ppump.size*fpump.size-I.size+1), constant_values=np.nan+0j).reshape(Ppump.size, fpump.size)
        plt2dimg_update(imgs[k], 20*np.log10(np.abs(S21[k]/refsignal[k])).T)
        # variance
        Ivar, Qvar = Ivarhandle.fetch_all()['value'], Qvarhandle.fetch_all()['value']
        refsignalvar[k] = Ivar[0] + 1j*Qvar[0]
        S21var[k] = np.pad(Ivar[1:]+1j*Qvar[1:], pad_width=(0,Ppump.size*fpump.size-Ivar.size+1), constant_values=np.nan+0j).reshape(Ppump.size, fpump.size)
    except Exception as e:
        print(repr(e))
    np.savez_compressed(
        fpath, Iflux=Iflux, Ppump=Ppump, fpump=fpump, Navg=Navg,
        refsignal=refsignal, refsignalvar=refsignalvar, S21=S21, S21var=S21var,
        tracetime=tracetime, fsignal=fsignal,
        config=config.meta)
    print("Time per trace:", (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime)), 's')

    fig.savefig(fpath+'.png', dpi=300)

    rfsource.write_str_with_opc(":output off")
    qm.octave.set_rf_output_mode(element, octave.RFOutputMode.off)

    # Plot SNR
    refsnr = np.nanmean(np.abs(refsignal)) / np.sqrt(np.nanmean(refsignalvar.real + refsignalvar.imag))
    nrows = int(np.sqrt(Iflux.size))
    ncols = int(np.ceil(Iflux.size/nrows))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, layout='constrained', squeeze=False, figsize=(12,8))
    imgs = [None] * Iflux.size
    axs[0,0].set_ylabel('Pump power / dBm')
    axs[-1,0].set_xlabel('Pump freq / GHz')
    for k, I in enumerate(Iflux):
        axs[k//ncols,k%ncols].set_title(f"Iflux {I*1e3:.5f} mA", fontsize=10)
        snr = np.abs(S21[k]) / np.sqrt(S21var[k].real + S21var[k].imag)
        imgs[k] = plt2dimg(
            axs[k//ncols,k%ncols], fpump/1e9, Ppump, snr.T,
            norm=CenteredNorm(vcenter=refsnr), cmap='coolwarm')
        fig.colorbar(imgs[k], ax=axs[k//ncols,k%ncols], orientation='horizontal', shrink=0.8)
    readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
    if element == 'vna':
        title = (
            f"signal {config.vnaLO/1e9:.5f}GHz+{config.vnaIF/1e6:.3f}MHz   Navg {Navg}"
            f"\n{config.readout_len/1e3}us readout at {readoutpower:.1f}dBm{config.vna_output_gain:+.1f}dB"
            f",   {config.input_gain:+.1f}dB input gain"
            f"\nColorbar:  amplitude SNR; Pump off SNR {refsnr:.1e}")
    else:
        title = (
            f"signal {config.resonatorLO/1e9:.5f}GHz+{config.resonatorIF/1e6:.3f}MHz   Navg {Navg}"
            f"\n{config.readout_len/1e3}us readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
            f",   {config.input_gain:+.1f}dB input gain"
            f"\nColorbar:  amplitude SNR; Pump off SNR {refsnr:.1e}")
    fig.suptitle(title, fontsize=10)
    fig.savefig(fpath+'_snr.png', dpi=300)

#%%

k = -1
fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True)
img = plt2dimg(axs[0], fpump/1e9, Ppump, np.abs(S21[k]/refsignal[k]).T)
fig.colorbar(img, ax=axs[0], label='signal gain')
img = plt2dimg(axs[1], fpump/1e9, Ppump, np.sqrt(S21var[k].real+S21var[k].imag).T)
fig.colorbar(img, ax=axs[1], label='noise')
img = plt2dimg(axs[2], fpump/1e9, Ppump, (np.abs(S21[k])/np.sqrt(S21var[k].real+S21var[k].imag)).T)
fig.colorbar(img, ax=axs[2], label='SNR')

#%%

plt.figure()
pidx = 0
plt.plot(S21[0,pidx,:].real, S21[0,pidx,:].imag, '.-', label=f"Pump {Ppump[pidx]:.1f}dBm")
plt.gca().add_patch(plt.Circle((S21[0,pidx,-1].real, S21[0,pidx,-1].imag), np.mean(np.sqrt(S21var[0,pidx,:].real + S21var[0,pidx,:].imag)), alpha=0.3, color='C0'))

pidx = 7
plt.plot(S21[0,pidx,:].real, S21[0,pidx,:].imag, '.-', label=f"Pump {Ppump[pidx]:.1f}dBm")
# for j in range(S21.shape[-1]):
#     plt.gca().add_patch(plt.Circle((S21[0,pidx,j].real, S21[0,pidx,j].imag), np.sqrt(S21var[0,pidx,j].real+S21var[0,pidx,j].imag), alpha=0.3, color='C0'))

plt.gca().add_patch(plt.Circle((refsignal.real, refsignal.imag), np.sqrt(refsignalvar.real), alpha=0.3, color='r'))
plt.plot(refsignal.real, refsignal.imag, 'd', color='r', label="ref signal")
print(np.sqrt(refsignalvar.real), np.sqrt(refsignalvar.imag))
print(np.sqrt(refsignalvar.real+refsignalvar.imag))
print(refsignalvar.real/refsignalvar.imag)
plt.legend()
plt.axis('equal')
