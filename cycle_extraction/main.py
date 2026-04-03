import logging

from config import PatientDir, PATHS
from cycle_extraction import fill_gaps_for_pdirs
from cycle_extraction.cycle_extraction_for_segments import cycle_extraction_for_pdirs
from utils.utils import FunctionTimer


def cycle_extraction(
        pdirs: list[PatientDir],
        ask_confirm: bool = True,
):
    if ask_confirm:
        print(f"Cycle Extraction for {PATHS.root}. Patients:")
        for pdir in pdirs:
            print(f"  {pdir.name}")
        input(f"Press enter to continue.")

    logging.info("---- Cycle Extraction ----")
    with FunctionTimer('Total Cycle Extraction'):
        logging.info("---- Filling Feature Gaps ----")
        with FunctionTimer('fill_feature_gaps'):
            fill_gaps_for_pdirs(pdirs)

        logging.info("---- Cycle Extraction for Segments ----")
        with FunctionTimer('cycle_extraction_for_segments'):
            cycle_extraction_for_pdirs(pdirs)


if __name__ == '__main__':
    pdirs_ = PATHS.patient_dirs()
    cycle_extraction(pdirs_)
