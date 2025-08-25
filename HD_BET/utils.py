import os
from glob import glob

join = os.path.join
isfile = os.path.isfile
isdir = os.path.isdir

def nifti_files(folder: str):
    return glob(join(folder, '*.nii.gz'))

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)