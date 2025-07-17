# Show the names of the data files while accessing the server remotely
import os

if __name__ == '__main__':
    path = r'/data/datasets/20240201_UNEEG_ForMayo/B52K3P3G/V5a'
    files = os.listdir(path)
    # files = os.listdir('.')
    print(files)