import logging

from scripts import copy_uneeg_data
from data_cleaning.main import data_cleaning
from preprocessing.main import preprocessing

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    # copy_uneeg_data.main()
    data_cleaning(ask_confirm=False)
    preprocessing(ask_confirm=False)


if __name__ == '__main__':
    main()
