
import time
import numpy as np
import importlib
import qm.qua as qua
import qm.octave as octave

from helpers import data_path

import configuration as config
#import configuration_RF1 as config
import qminit

#qmm = qminit.connect(use_calibration=False)
qmm = qminit.connect()
from qualang_tools.loops import from_array


#%%

from RsInstrument import RsInstrument

fsva = RsInstrument('TCPIP::169.254.0.7::INSTR', id_query=True, reset=False)
fsva.visa_timeout = 1000


#%%

filename = '{datetime}_fsva_octave_with_2x_VLF3000'
fpath = data_path(filename)

fLOs = np.array([2e9, 2.1e9, 2.2e9, 2.3e9, 2.4e9, 2.45e9, 2.5e9, 2.6e9, 2.7e9, 2.8e9, 2.9e9, 3.0e9, 3.01e9, 3.1e9, 3.2e9])
fIFs = np.array([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]) * 1e6

# FSVA meta
channel = 1
meta = {
	'center': float(fsva.query(f'SENS{channel}:FREQ:CENT?')),
	'span': float(fsva.query(f'SENS{channel}:FREQ:SPAN?')),
	'start': float(fsva.query(f'SENS{channel}:FREQ:STAR?')),
	'stop': float(fsva.query(f'SENS{channel}:FREQ:STOP?')),
	'points': int(fsva.query(f'SENS{channel}:SWEep:POINts?')),
	'bandwidth': float(fsva.query(f'SENS{channel}:BAND?')),
    'videobandwidth': float(fsva.query(f'SENS{channel}:BAND:VIDEO?')),
	'sweep_type': fsva.query(f'SENS{channel}:SWE:TYPE?'),
	'sweep_time': float(fsva.query(f'SENS{channel}:SWEep:TIME?')),
}

freq = np.array(fsva.query_bin_or_ascii_float_list(":TRACE:DATA:X? TRACE1"))
spectra = np.full((len(fLOs), len(fIFs), len(freq)), np.nan)

# QM program
importlib.reload(config)
importlib.reload(qminit)
QMSLEEP = 0.01

try:
    for i, fLO in enumerate(fLOs):
        config.qubitLO = fLO
        config.qmconfig['elements']['qubit']['mixInputs']['lo_frequency'] = fLO
        config.qmconfig['mixers']['octave_octave1_2'][0]['lo_frequency'] = fLO
        qm = qmm.open_qm(config.qmconfig)
        qminit.octave_setup_qubit(qm, config)
    
        with qua.program() as mixer_cal_qubit:
            f = qua.declare(int)
            with qua.for_(*from_array(f, fIFs)):
                qua.update_frequency('qubit', f)
                qua.pause()
                qua.play('const', 'qubit', duration=250000000)
        job = qm.execute(mixer_cal_qubit)

        for j, fIF in enumerate(fIFs):
            while not job.is_paused():
                time.sleep(QMSLEEP)
            print(f"{fLO/1e9:.2f}GHz {fIF/1e6:+.1f}MHz")
            job.resume()
            time.sleep(0.5)
            spectra[i,j] = np.array(fsva.query_bin_or_ascii_float_list(":TRACE:DATA? TRACE1"))
except Exception as e:
    job.halt()
    raise e
finally:
    np.savez(fpath, freq=freq, spectra=spectra,
             meta=meta, config=config.meta)



from helpers import plt2dimg
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

y = np.arange(fLOs.size*fIFs.size)
fSB = np.array([fLO + fIF for fLO in fLOs for fIF in fIFs])

fig, axs = plt.subplots(figsize=(15, 6), constrained_layout=True)#layout='constrained')
img = plt2dimg(axs, freq/1e9, y, spectra.reshape((-1, freq.size)).T)
fig.colorbar(img).set_label("Octave output / dBm")
#axs.scatter(fSB/1e9, y, color='r', s=1)

axs.set_xlabel('Frequency / GHz')
axs.set_ylabel('Measurement number')

bwstr = EngFormatter(sep="").format_eng(meta['bandwidth']) + "Hz"
vbwstr = EngFormatter(sep="").format_eng(meta['videobandwidth']) + "Hz"
fig.suptitle(
    f"FSVA  ResBW {bwstr}  VideoBW {vbwstr}\n"
    f"const pulse  {10*np.log10(config.const_amp**2 * 10):+.1f}dBm OPX {config.qubit_output_gain:+.1f}dB Octave")
#plt.tight_layout()
plt.savefig(fpath+'.png', dpi=300)
