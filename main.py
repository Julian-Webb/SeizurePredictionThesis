import logging

from data_cleaning.main import data_cleaning
from scripts import copy_uneeg_data

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    copy_uneeg_data.main()
    data_cleaning(ask_confirm=False)


if __name__ == '__main__':
    main()
