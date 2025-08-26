import os.path
from multiprocessing import Pool
import sys
import SimpleITK as sitk
import torch
import numpy as np


sys.stdout = open(os.devnull, 'w')
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
sys.stdout = sys.__stdout__
from HD_BET.paths import folder_with_parameter_files
from .utils import nifti_files, maybe_mkdir_p, isdir, join


def apply_bet(
    img: str,
    bet: str,
    out_fname: str
):
    img_itk = sitk.ReadImage(img)
    img_npy = sitk.GetArrayFromImage(img_itk)
    img_bet = sitk.GetArrayFromImage(sitk.ReadImage(bet))
    img_npy[img_bet == 0] = 0
    out = sitk.GetImageFromArray(img_npy)
    out.CopyInformation(img_itk)
    sitk.WriteImage(out, out_fname)


def get_hdbet_predictor(
    use_tta: bool = False,
    device: torch.device = torch.device('mps'),
    verbose: bool = False
):
    os.environ['nnUNet_compile'] = 'F'
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose
    )
    predictor.initialize_from_trained_model_folder(
        folder_with_parameter_files,
        'all'
    )
    if device == torch.device('cpu'):
        torch.set_num_threads(os.cpu_count())
    return predictor


def hdbet_predict(
    input_file_or_folder: str,
    output_file_or_folder: str,
    predictor: nnUNetPredictor = None,
    keep_brain_mask: bool = False,
    compute_brain_extracted_image: bool = True,
    predictor_kwargs: dict = None
):
    # find input file or files
    if os.path.isdir(input_file_or_folder):
        input_files = nifti_files(input_file_or_folder)
        # output_file_or_folder must be folder in this case
        maybe_mkdir_p(output_file_or_folder)
        output_files = [join(output_file_or_folder, os.path.basename(i)) for i in input_files]
        brain_mask_files = [i[:-7] + '_bet.nii.gz' for i in output_files]
    else:
        assert not isdir(output_file_or_folder), 'If input is a single file then output must be a filename, not a directory'
        assert output_file_or_folder.endswith('.nii.gz'), 'Output file must end with .nii.gz'
        input_files = [input_file_or_folder]
        output_files = [join(os.path.curdir, output_file_or_folder)]
        brain_mask_files = [join(os.path.curdir, output_file_or_folder[:-7] + '_bet.nii.gz')]
    
    if predictor is None:
        assert predictor_kwargs is not None
        predictor = get_hdbet_predictor(**predictor_kwargs)

    # we first just predict the brain masks using the standard nnU-Net inference
    predictor.predict_from_files(
        [[i] for i in input_files],
        brain_mask_files,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=8,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    # remove unnecessary json files
    os.remove(join(os.path.dirname(brain_mask_files[0]), 'dataset.json'))
    os.remove(join(os.path.dirname(brain_mask_files[0]), 'plans.json'))
    os.remove(join(os.path.dirname(brain_mask_files[0]), 'predict_from_raw_data_args.json'))

    if compute_brain_extracted_image:
        # now brain extract the images
        res = []
        with Pool(4) as p:
            for im, bet, out in zip(input_files, brain_mask_files, output_files):
                res.append(
                    p.starmap_async(
                    apply_bet,
                    ((im, bet, out),)
                    )
                )
            [i.get() for i in res]

    if not keep_brain_mask:
        [os.remove(i) for i in brain_mask_files]


def hdbet_predict_return_array(
    image,
    predictor = None,
    predictor_kwargs: dict = None,
    num_proceses_preprocessing: int = 3,
    num_processes_segmentation_export: int = 3
) -> np.ndarray:

    if predictor is None:
        assert predictor_kwargs is not None
        predictor = get_hdbet_predictor(**predictor_kwargs)

    brainseg = predictor.predict_from_files(
        [[image]],
        None,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=num_proceses_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    return brainseg[0]
