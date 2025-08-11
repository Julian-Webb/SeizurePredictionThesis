# Show the names of the data files while accessing the server remotely
import os

if __name__ == '__main__':
    print(f'{os.getcwd()=}')
    data_folder = r'/Users/julian/Developer/EEG Data/'
    # data_folder = r'/data/datasets/'
    path = os.path.join(data_folder, r'20240201_UNEEG_ForMayo/B52K3P3G')
    files = os.listdir(path)
    print(f'Files in {path}:')
    print(files)