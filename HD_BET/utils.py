import os
from glob import glob


join = os.path.join
isfile = os.path.isfile
isdir = os.path.isdir
basename = os.path.basename
dirname = os.path.dirname


def nifti_files(folder: str):
    return glob(join(folder, '*.nii.gz'))

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def get_default_device() -> str:
    import torch
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    return device