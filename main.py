import logging

from scripts import copy_uneeg_data
from data_cleaning.main import data_cleaning
from preprocessing.main import preprocessing


def main(copy_data: bool):
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    if copy_data:
        copy_uneeg_data.main()
    data_cleaning(ask_confirm=False)
    preprocessing(ask_confirm=False)


if __name__ == '__main__':
    main(copy_data=False)
