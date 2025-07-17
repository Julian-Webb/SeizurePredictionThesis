# Show the names of the data files while accessing the server remotely
import os

if __name__ == '__main__':
    print(f'{os.getcwd()=}')
    path = r'/data/datasets/20240201_UNEEG_ForMayo/B52K3P3G/V5a'
    files = os.listdir(path)
    print(f'Files in {path}:')
    print(files)