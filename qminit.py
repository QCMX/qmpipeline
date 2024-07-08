#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup of quantum machine and octave.
Contains information about setup, because the OPX setup is in the config file,
but the octave setup is set in a different API style :/

Mostly uses additionnal values from configuration file,
but eg. external vs internal LO and which RF input port to use are only here.
"""

import os
import qm.octave as octave
from qm.QuantumMachinesManager import QuantumMachinesManager


def connect(use_calibration=True):
    opx_ip = '169.254.0.10'
    opx_port = 9510
    octave_ip = '169.254.0.12'
    octave_port = 50

    octave_config = octave.QmOctaveConfig()
    octave_config.add_device_info('octave1', octave_ip, octave_port)
    octave_config.set_opx_octave_mapping([('con1', 'octave1')]) # standard port mapping
    if use_calibration:
        octave_config.set_calibration_db(os.getcwd())

    qmm = QuantumMachinesManager(host=opx_ip, port=opx_port, octave=octave_config)
    return qmm


def octave_setup_resonator(qm, config, short_readout_gain=False):
    element = 'resonator'
    outgain = config.resonator_output_gain+(config.short_readout_amp_gain if short_readout_gain else 0)
    qm.octave.set_lo_source(element, octave.OctaveLOSource.Internal)
    qm.octave.set_lo_frequency(element, config.resonatorLO)
    qm.octave.set_rf_output_gain(element, outgain)
    qm.octave.set_rf_output_mode(element, octave.RFOutputMode.on)

    qm.octave.set_qua_element_octave_rf_in_port(element, "octave1", 1)
    qm.octave.set_downconversion(
        element, lo_source=octave.RFInputLOSource.Internal, lo_frequency=config.resonatorLO,
        if_mode_i=octave.IFMode.direct, if_mode_q=octave.IFMode.direct
    )

def octave_setup_vna(qm, config):
    element = 'vna'
    outgain = config.vna_output_gain
    qm.octave.set_lo_source(element, octave.OctaveLOSource.Internal)
    qm.octave.set_lo_frequency(element, config.vnaLO)
    qm.octave.set_rf_output_gain(element, outgain)
    qm.octave.set_rf_output_mode(element, octave.RFOutputMode.on)

    qm.octave.set_qua_element_octave_rf_in_port(element, "octave1", 1)
    qm.octave.set_downconversion(
        element, lo_source=octave.RFInputLOSource.Internal, lo_frequency=config.vnaLO,
        if_mode_i=octave.IFMode.direct, if_mode_q=octave.IFMode.direct
    )

def octave_setup_qubit(qm, config):
    element = 'qubit'
    qm.octave.set_lo_source(element, octave.OctaveLOSource.Internal)
    #qm.octave.set_lo_source(element, octave.OctaveLOSource.LO2)
    qm.octave.set_lo_frequency(element, config.qubitLO)
    qm.octave.set_rf_output_gain(element, config.qubit_output_gain)
    qm.octave.set_rf_output_mode(element, octave.RFOutputMode.on)
