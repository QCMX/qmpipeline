#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup of quantum machine and octave.
Contains information about setup, because the OPX setup is in the config file,
but the octave setup is set in a different API style :/

Mostly uses additionnal values from configuration file,
but eg. external vs internal LO and which RF input port to use are only here.

Works with qua-1.2.6.
"""

import os
import qm.octave as octave
from qm.quantum_machines_manager import QuantumMachinesManager


def connect(
        use_calibration=True,
        opx_ip='169.254.0.10',
        octave_ip='169.254.0.12',
        octave_port=50):
    opx_port = 9510

    octave_config = octave.QmOctaveConfig()
    octave_config.add_device_info('octave1', octave_ip, octave_port)

    qmm = QuantumMachinesManager(
        host=opx_ip, port=opx_port,
        octave=octave_config,
        octave_calibration_db_path=os.getcwd() if use_calibration else None)
    return qmm
