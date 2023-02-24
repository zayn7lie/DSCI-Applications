import os

def workDirChanger(path):
    os.chdir(path)
    print('current working directory is {}'.format(os.getcwd()))

    print(os.getcwd())
    print(os.listdir(os.getcwd()))