import os
import shutil


def clean_dir(path: os.PathLike):
    try:
        shutil.rmtree(path)
    except:
        pass

    os.mkdir(path)
