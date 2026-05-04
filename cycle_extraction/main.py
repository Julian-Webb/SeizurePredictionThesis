import logging

from config import PatientDir, PATHS
from cycle_extraction import fill_gaps_for_pdirs
from cycle_extraction.cycle_extraction_for_segments import cycle_extraction_for_pdirs
from cycle_extraction.cycle_extraction_plots import cycle_extraction_plots_for_pdirs
from cycle_extraction.qualifications import qualify_model_features
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
        with FunctionTimer('cycle_extraction_for_pdirs'):
            cycle_extraction_for_pdirs(pdirs)
        with FunctionTimer('cycle_extraction_plots_for_pdirs'):
            cycle_extraction_plots_for_pdirs(pdirs)

        logging.info("---- Qualification of Models and Features ----")
        with FunctionTimer('qualify_model_features'):
            qualify_model_features()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs_ = PATHS.patient_dirs()
    cycle_extraction(pdirs_)
