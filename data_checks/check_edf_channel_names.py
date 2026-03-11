import logging
import multiprocessing
from typing import List

from pyedflib import EdfReader

from config.constants import CHANNELS
from config.paths import PatientDir, PATHS


def check_ptnt_edf_channel_names(pdir: PatientDir):
    """
    Make sure that all EDF files have the same channels in the same order.
    :param pdir:
    :return:
    """
    # Iterate over all EDFs and make sure the channel names are correct
    logging.info(f"Checking {pdir.name}")
    for edf_path in pdir.edf_dir.iterdir():
        channels = EdfReader(str(edf_path)).getSignalLabels()
        if channels != CHANNELS:
            logging.error(f"Channel names don't match {channels} for {edf_path.name}")
        else:
            logging.debug(f"Channel names match {channels} for {edf_path.name}")
    logging.info(f"Finished {pdir.name}")


def check_edf_channel_names(pdirs: List[PatientDir]):
    with multiprocessing.Pool() as pool:
        pool.map(check_ptnt_edf_channel_names, pdirs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    check_edf_channel_names(PATHS.patient_dirs(include_invalid_ptnts=True))
