import functools as fct
import itertools as itt
import os
from typing import Optional, Tuple, Union

import cv2
import dask as da
import dask.array as darr
import networkx as nx
import numpy as np
import pandas as pd
import sparse
import xarray as xr
import scipy.sparse 
import numba as nb
import SimpleITK as sitk
from typing import Optional, Tuple
import community
from sklearn.cluster import SpectralClustering
from distributed import get_client
from scipy.ndimage import median_filter
from scipy.ndimage import label
from scipy.signal import butter, lfilter
from scipy.stats import kstest, zscore
from skimage.morphology import disk
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree, radius_neighbors_graph
from typing import Callable, List, Optional, Union
from scipy.sparse.linalg import lsqr
import pyfftw.interfaces.numpy_fft as numpy_fft
import re
import zarr
import zarr as zr
from os import listdir
from os.path import isdir, isfile
from os.path import join as pjoin
from pathlib import Path
from uuid import uuid4
import rechunker
import shutil
import dask.array as darr
from dask.delayed import optimize as default_delay_optimize
from dask.optimization import cull, fuse, inline, inline_functions
from scipy.ndimage import uniform_filter
from medpy.filter.smoothing import anisotropic_diffusion
import SimpleITK as sitk
from skimage.registration import phase_cross_correlation
import warnings
from dask.core import flatten
from natsort import natsorted
from tifffile import TiffFile, imread
import ffmpeg
from copy import deepcopy
import _operator
from skimage import morphology as moph

FAST_FUNCTIONS = [
    darr.core.getter_inline,
    darr.core.getter,
    _operator.getitem,
    zr.core.Array,
    darr.chunk.astype,
    darr.core.concatenate_axes,
    darr.core._vindex_slice,
    darr.core._vindex_merge,
    darr.core._vindex_transpose,
]

def set_file_path(file_path):
    global dpath
    dpath=file_path

def denoise_by_frame(frame, method='gaussian', ksize=5):
    """
    Apply denoising to a single frame.

    Args:
        frame (np.array): The input frame.
        method (str): Denoising method ('gaussian', 'median', or 'bilateral').
        kernel_size (int): Size of the kernel used for denoising.

    Returns:
        np.array: The denoised frame.
    """
    kernel_size = int(ksize)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if method == 'gaussian':
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size),  sigmaX=0)
    elif method == 'median':
        return cv2.medianBlur(frame, kernel_size)
    elif method == 'bilateral':
        return cv2.bilateralFilter(frame, kernel_size, 75, 75)
    else:
        raise ValueError(f"Denoise method {method} not understood")
    
def remove_background_by_frame(frame, method="uniform", kernel_size=5):
    """
    Remove background from a single frame using specified method.

    Args:
        frame (np.array): The input frame.
        method (str): Background removal method ('uniform' or 'tophat').
        kernel_size (int): Size of the kernel used for background removal.

    Returns:
        np.array: Frame with background removed.
    """
    if method == "uniform":
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        background = cv2.filter2D(frame, -1, kernel)
        return frame - background
    elif method == "tophat":
        selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, selem)
    else:
        raise ValueError("Unknown background removal method: " + method)


     
def denoise(varr: xr.DataArray, method: str, **kwargs) -> xr.DataArray:
    """
    Denoise the movie frame by frame.

    This function wraps around several image processing functions to denoise the
    data frame by frame. All additional keyword arguments will be passed
    directly to the underlying functions.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data, should have dimensions "height", "width" and
        "frame".
    method : str
        The method to use to denoise each frame. If `"gaussian"`, then a
        gaussian filter will be applied using :func:`cv2.GaussianBlur`. If
        `"anisotropic"`, then anisotropic filtering will be applied using
        :func:`medpy.filter.smoothing.anisotropic_diffusion`. If `"median"`,
        then a median filter will be applied using :func:`cv2.medianBlur`. If
        `"bilateral"`, then a bilateral filter will be applied using
        :func:`cv2.bilateralFilter`.

    Returns
    -------
    res : xr.DataArray
        The resulting denoised movie. Same shape as input `varr` but will have
        `"_denoised"` appended to its name.

    Raises
    ------
    NotImplementedError
        if the supplied `method` is not recognized
    """
    if method == "gaussian":
        func = cv2.GaussianBlur
        if 'sigmaX' not in kwargs:
            kwargs['sigmaX'] = 0  # Default value for sigmaX
            kwargs['ksize']=(kwargs['ksize'], kwargs['ksize'])
    elif method == "anisotropic":
        func = anisotropic_diffusion
    elif method == "median":
        func = cv2.medianBlur
    elif method == "bilateral":
        func = cv2.bilateralFilter
    else:
        raise NotImplementedError("denoise method {} not understood".format(method))
    res = xr.apply_ufunc(
        func,
        varr,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=kwargs,  # Pass kwargs directly
    )
    res = res.astype(varr.dtype)
    return res.rename(varr.name + "_denoised")

def remove_background(varr: xr.DataArray, method: str, wnd: int) -> xr.DataArray:
    """
    Remove background from a video.

    This function remove background frame by frame. Two methods are available
    for use: if `method == "uniform"`, then the background is estimated by
    convolving the frame with a uniform/mean kernel and then subtract it from
    the frame. If `method == "tophat"`, then a morphological tophat operation is
    applied to each frame.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data, should have dimensions "height", "width" and
        "frame".
    method : str
        The method used to remove the background. Should be either `"uniform"`
        or `"tophat"`.
    wnd : int
        Window size of kernels used for background removal, specified in pixels.
        If `method == "uniform"`, this will be the size of a box kernel
        convolved with each frame. If `method == "tophat"`, this will be the
        radius of a disk kernel used for morphological operations.

    Returns
    -------
    res : xr.DataArray
        The resulting movie with background removed. Same shape as input `varr`
        but will have `"_subtracted"` appended to its name.

    See Also
    --------
    `Morphology <https://docs.opencv.org/4.5.2/d9/d61/tutorial_py_morphological_ops.html>`_ :
        for details about morphological operations
    """
    selem = disk(wnd)
    res = xr.apply_ufunc(
        remove_background_perframe,
        varr,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=dict(method=method, wnd=wnd, selem=selem),
    )
    res = res.astype(varr.dtype)
    return res.rename(varr.name + "_subtracted")


def remove_background_perframe(
    fm: np.ndarray, method: str, wnd: int, selem: np.ndarray
) -> np.ndarray:
    """
    Remove background from a single frame.

    Parameters
    ----------
    fm : np.ndarray
        The input frame.
    method : str
        Method to use to remove background. Should be either `"uniform"` or
        `"tophat"`.
    wnd : int
        Size of the uniform filter. Only used if `method == "uniform"`.
    selem : np.ndarray
        Kernel used for morphological operations. Only used if `method == "tophat"`.

    Returns
    -------
    fm : np.ndarray
        The frame with background removed.

    See Also
    --------
    remove_background : for detailed explanations
    """
    if method == "uniform":
        return fm - uniform_filter(fm, wnd)
    elif method == "tophat":
        return cv2.morphologyEx(fm, cv2.MORPH_TOPHAT, selem)



def estimate_motion(
    varr: xr.DataArray, dim="frame", npart=3, chunk_nfm: Optional[int] = None, **kwargs
) -> xr.DataArray:
    """
    Estimate motion for each frame of the input movie data.

    This function estimates motion using a recursive approach. The movie is
    splitted into chunks of `npart` frames and motion estimation is carried out
    within each chunk relative to the middle frame, then a template is generated
    for each chunk by aggregating the motion-corrected frames within each chunk.
    Next, every `npart` chunks are grouped together, and motion estimation is
    carried out within each group relative to the middle chunk using the
    aggregated templates. The chunk-level motions are added on top of the
    previous within-chunk level motions. This step is then repeated recursively
    until we are left with a single chunk representing the full movie, at which
    point the motion estimation is finished.

    The motion estimation itself is carried out with fft-based phase correlation
    by default. Alternatively, non-rigid motion correction can be carried out by
    modelling the motion of each frame as translations of individual vertices of
    a smooth BSpline mesh. The estimation of the translations can then be find
    by gradient descent using correlation between frames as objective. This
    feature is currently experimental. Additionally, various correction
    procedures can be carry out to filter out frames not suited as template for
    motion correction, or to correct for large false shifts when the quality of
    templates are low.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data.
    dim : str, optional
        The dimension along which motion estimation should be carried out. By
        default `"frame"`.
    npart : int, optional
        Number of frames/chunks to combine for the recursive algorithm. By
        default `3`.
    chunk_nfm : int, optional
        Number of frames in each parallel task. Note that this only affects dask
        graph construction, but not the recursion of the algorithm. If `None`
        then the dask chunksize along `dim` will be used. By default `None`.

    Keyword Arguments
    -----------------
    alt_error : float, optional
        Error threshold between estimated shifts from two alternative methods,
        specified in pixels. If not `None`, then for each chunk during
        recursion, the first and last frame of that chunk will be returned in
        addition to the aggregated template. And when estimating motion between
        chunks, the estimation will be carried out twice: once using the
        aggregated templates, once using the consecutive first/last frames
        between chunks. The result of these two methods will then be compared.
        If their absolute difference is larger than `alt_error`, then the result
        with smaller absolute value (closer to zero shifts) will be used. This
        is useful to correct for cases where activities of cells are sparse and
        changing across chunks, leading to wrong features being matched in
        aggregated templates. If `None` then no additional checking will be
        performed. By default `5`.
    aggregation : str, optional
        How frames should be aggregated to generate the template for each chunk.
        Should be either "mean" or "max". By default `"mean"`.
    upsample : int, optional
        The upsample factor passed to
        :func:`skimage.registration.phase_cross_correlation` to achieve
        sub-pixel accuracy.
    circ_thres : float, optional
        The circularity threshold to check whether a frame can serve as a good
        template for estimating motion. If not `None`, then for each frame a
        comparison image is computed using :func:`cv2.matchTemplate` between the
        frame and zero-padded version (up to `max_sh`) using
        `cv2.TM_SQDIFF_NORMED`. The comparison image of a good template should
        only have `< 1` values around the center and the `< 1` region should be
        circular. Hence the circularity defined as `4 * np.pi * (area /
        (perimeter ** 2))` for the `< 1` region is computed, and any frame with
        circularity smaller than `circ_thres` is excluded from propagation of
        shifts and aggregation of templates. By default `None`.
    max_sh : int, optional
        Amount of zero padding when checking for the quality of frames,
        specified in pixels. Only used if `circ_thres is not None`. See
        `circ_thres` for more detail. By default `100`.
    mesh_size : Tuple[int, int], optional
        Number of control points for the BSpline mesh in each dimension,
        specified in the order ("height", "width"). If not `None` then the
        experimental non-rigid motion estimation is enabled. By default `None`
    niter : int, optional
        Max number of iteration for the gradient descent process of estimation
        BSpline parameters. Only used if `mesh_size is not None`. By default
        `100`.
    bin_thres : int, optional
        Intensity threshold for binarizing the frames. The binarized frame will
        be used as masks for non-rigid motion estimation, where only pixels in
        the mask will be used to evaluate the gradient during optimization.
        Significantly improve performance but sacrifice accuracy of estimation
        for dim regions. Only used if `mesh_size is not None`. By default
        `None`.

    Returns
    -------
    motion : xr.DataArray
        Estimated motion for each frame. Has two dimensions `dim` and
        `"shift_dim"` representing rigid shifts in each direction if `mesh_size
        is None`, otherwise has four dimensions: `dim`, `"grid0"`, `"grid1"` and
        `"shift_dim"` representing shifts for each mesh grid control point.

    See Also
    --------
    :doc:`simpleitk:registrationOverview` :
        for overview of the non-rigid estimation method
    """
    varr = varr.transpose(..., dim, "height", "width")
    loop_dims = list(set(varr.dims) - set(["height", "width", dim]))
    if npart is None:
        # by default use a npart that result in two layers of recursion
        npart = max(3, int(np.ceil((varr.sizes[dim] / chunk_nfm) ** (1 / 2))))
    if loop_dims:
        loop_labs = [varr.coords[d].values for d in loop_dims]
        res_dict = dict()
        for lab in itt.product(*loop_labs):
            va = varr.sel({loop_dims[i]: lab[i] for i in range(len(loop_dims))})
            vmax, sh = est_motion_part(va.data, npart, chunk_nfm, **kwargs)
            if kwargs.get("mesh_size", None):
                sh = xr.DataArray(
                    sh,
                    dims=[dim, "shift_dim", "grid0", "grid1"],
                    coords={
                        dim: va.coords[dim].values,
                        "shift_dim": ["height", "width"],
                    },
                )
            else:
                sh = xr.DataArray(
                    sh,
                    dims=[dim, "shift_dim"],
                    coords={
                        dim: va.coords[dim].values,
                        "shift_dim": ["height", "width"],
                    },
                )
            res_dict[lab] = sh.assign_coords(**{k: v for k, v in zip(loop_dims, lab)})
        sh = xrconcat_recursive(res_dict, loop_dims)
    else:
        vmax, sh = est_motion_part(varr.data, npart, chunk_nfm, **kwargs)
        if kwargs.get("mesh_size", None):
            sh = xr.DataArray(
                sh,
                dims=[dim, "shift_dim", "grid0", "grid1"],
                coords={
                    dim: varr.coords[dim].values,
                    "shift_dim": ["height", "width"],
                },
            )
        else:
            sh = xr.DataArray(
                sh,
                dims=[dim, "shift_dim"],
                coords={
                    dim: varr.coords[dim].values,
                    "shift_dim": ["height", "width"],
                },
            )
    return sh

def custom_fused_keys_renamer(
    keys: list, max_fused_key_length=120, rename_dict: Optional[dict] = None
) -> str:
    """
    Custom implmentation to create new keys for `fuse` tasks.

    Uses custom `split_key` implementation.

    Parameters
    ----------
    keys : list
        List of task keys that should be fused together.
    max_fused_key_length : int, optional
        Used to limit the maximum string length for each renamed key. If `None`,
        there is no limit. By default `120`.
    rename_dict : dict, optional
        Dictionary used to rename keys during fuse. By default `None`.

    Returns
    -------
    fused_key : str
        The fused task key.

    See Also
    -------
    split_key
    dask.optimization.fuse
    """
    it = reversed(keys)
    first_key = next(it)
    typ = type(first_key)

    if max_fused_key_length:  # Take into account size of hash suffix
        max_fused_key_length -= 5


def est_motion_perframe(
    src: np.ndarray,
    dst: np.ndarray,
    upsample: int,
    src_ma: Optional[np.ndarray] = None,
    dst_ma: Optional[np.ndarray] = None,
    mesh_size: Optional[Tuple[int, int]] = None,
    niter=100,
) -> np.ndarray:
    """
    Estimate motion given two frames.

    Parameters
    ----------
    src : np.ndarray
        The frame to be registered.
    dst : np.ndarray
        The destination frame of registration.
    upsample : int
        Upsample factor.
    src_ma : np.ndarray, optional
        Boolean mask for `src`. Only used if `mesh_size is not None`. By default
        `None`.
    dst_ma : np.ndarray, optional
        Boolean mask for `dst`. Only used if `mesh_size is not None`. By default
        `None`.
    mesh_size : Tuple[int, int], optional
        Number of control points for the BSpline mesh in each dimension. By
        default `None`.
    niter : int, optional
        Max number of iteration for the gradient descent process. By default
        `100`.

    Returns
    -------
    motion : np.ndarray
        Estimated motion between two frames.

    See Also
    --------
    estimate_motion : for detailed explanation of parameters
    """
    sh, err, dif= phase_cross_correlation(
        src,
        dst,
        upsample_factor=upsample,
        return_error=False,
    )
    if mesh_size is None:
        return -sh
    src = sitk.GetImageFromArray(src.astype(np.float32))
    dst = sitk.GetImageFromArray(dst.astype(np.float32))
    reg = sitk.ImageRegistrationMethod()
    sh = sh[::-1]
    trans_init = sitk.TranslationTransform(2, sh)
    reg.SetMovingInitialTransform(trans_init)
    if src_ma is not None:
        reg.SetMetricMovingMask(sitk.GetImageFromArray(src_ma.astype(np.uint8)))
    if dst_ma is not None:
        reg.SetMetricFixedMask(sitk.GetImageFromArray(dst_ma.astype(np.uint8)))
    trans_opt = sitk.BSplineTransformInitializer(
        image1=dst, transformDomainMeshSize=mesh_size
    )
    reg.SetInitialTransform(trans_opt)
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.1, convergenceMinimumValue=1e-5, numberOfIterations=niter
    )
    tx = reg.Execute(dst, src)
    coef = np.stack(
        [sitk.GetArrayFromImage(im) for im in tx.Downcast().GetCoefficientImages()]
    )
    coef = coef + sh.reshape((2, 1, 1))
    return coef

def est_motion_part(
    varr: darr.Array, npart: int, chunk_nfm: int, alt_error=5, **kwargs
) -> Tuple[darr.Array, darr.Array]:
    """
    Construct dask graph for the recursive motion estimation algorithm.

    Parameters
    ----------
    varr : darr.Array
        Input dask array representing movie data.
    npart : int
        Number of frames/chunks to combine for the recursive algorithm.
    chunk_nfm : int
        Number of frames in each parallel task.
    alt_error : int, optional
        Error threshold between estimated shifts from two alternative methods,
        specified in pixels. By default `5`.

    Returns
    -------
    temps : darr.Array
        Registration template for the movie.
    shifts : darr.Array
        Estimated motion.
    See Also
    --------
    estimate_motion
    """
    if chunk_nfm is None:
        chunk_nfm = varr.chunksize[0]
    varr = varr.rechunk((chunk_nfm, None, None))
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^est_motion_chunk"])
    if kwargs.get("mesh_size", None):
        param = get_bspline_param(varr[0].compute(), kwargs["mesh_size"])
    tmp_ls = []
    sh_ls = []
    for blk in varr.blocks:
        res = da.delayed(est_motion_chunk)(
            blk, None, alt_error=alt_error, npart=npart, **kwargs
        )
        if alt_error:
            tmp = darr.from_delayed(
                res[0], shape=(3, blk.shape[1], blk.shape[2]), dtype=blk.dtype
            )
        else:
            tmp = darr.from_delayed(
                res[0], shape=(blk.shape[1], blk.shape[2]), dtype=blk.dtype
            )
        if kwargs.get("mesh_size", None):
            sh = darr.from_delayed(
                res[1],
                shape=(blk.shape[0], 2, int(param[1]), int(param[0])),
                dtype=float,
            )
        else:
            sh = darr.from_delayed(res[1], shape=(blk.shape[0], 2), dtype=float)
        tmp_ls.append(tmp)
        sh_ls.append(sh)
    with da.config.set(array_optimize=arr_opt):
        temps = da.optimize(darr.stack(tmp_ls, axis=0))[0]
        shifts = da.optimize(darr.concatenate(sh_ls, axis=0))[0]
    while temps.shape[0] > 1:
        tmp_ls = []
        sh_ls = []
        for idx in np.arange(0, temps.numblocks[0], npart):
            tmps = temps.blocks[idx : idx + npart]
            sh_org = shifts.blocks[idx : idx + npart]
            sh_org_ls = [sh_org.blocks[i] for i in range(sh_org.numblocks[0])]
            res = da.delayed(est_motion_chunk)(
                tmps, sh_org_ls, alt_error=alt_error, npart=npart, **kwargs
            )
            if alt_error:
                tmp = darr.from_delayed(
                    res[0], shape=(3, tmps.shape[1], tmps.shape[2]), dtype=tmps.dtype
                )
            else:
                tmp = darr.from_delayed(
                    res[0], shape=(tmps.shape[1], tmps.shape[2]), dtype=tmps.dtype
                )
            sh_new = darr.from_delayed(res[1], shape=sh_org.shape, dtype=sh_org.dtype)
            tmp_ls.append(tmp)
            sh_ls.append(sh_new)
        temps = darr.stack(tmp_ls, axis=0)
        shifts = darr.concatenate(sh_ls, axis=0)
    return temps, shifts

def est_motion_chunk(
    varr: np.ndarray,
    sh_org: np.ndarray,
    npart: int,
    alt_error: float,
    aggregation="mean",
    upsample=100,
    max_sh=100,
    circ_thres: Optional[float] = None,
    mesh_size: Optional[Tuple[int, int]] = None,
    niter=100,
    bin_thres: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carry out motion estimation per chunk.

    Parameters
    ----------
    varr : np.ndarray
        Input chunk of movie.
    sh_org : np.ndarray
        Original motions to be added upon.
    npart : int
        Number of frames/chunks to combine for the recursive algorithm.
    alt_error : float
        Error threshold between estimated shifts from two alternative methods,
        specified in pixels.
    aggregation : str, optional
        How frames should be aggregated to generate the template. By default
        `"mean"`.
    upsample : int, optional
        The upsample factor. By default `100`.
    max_sh : int, optional
        Amount of zero padding when checking for the quality of frames,
        specified in pixels. By default `100`.
    circ_thres : float, optional
        The circularity threshold to check whether a frame can serve as a good
        template for estimating motion. By default `None`.
    mesh_size : Tuple[int, int], optional
        Number of control points for the BSpline mesh in each dimension. By
        default `None`.
    niter : int, optional
        Max number of iteration for the gradient descent process. By default `100`.
    bin_thres : float, optional
        Intensity threshold for binarizing the frames. By default `None`.

    Returns
    -------
    tmp : np.ndarray
        The template of current chunk for further motion estimation.
    motions : np.ndarray
        Motions between frames within the chunk.

    Raises
    ------
    ValueError
        if `aggregation` is not `"mean"` or `"max"`

    See Also
    --------
    estimate_motion : for detailed explanation of parameters
    """
    if varr.ndim == 3 and varr.shape[0] == 1:
        if sh_org is not None:
            motions = sh_org
        else:
            if mesh_size is not None:
                # TODO handle non-rigid case
                pass
            else:
                motions = np.array([0, 0])[np.newaxis, :]
        if alt_error:
            tmp = np.stack([varr[0]] * 3)
        else:
            tmp = varr[0]
        return tmp, motions
    while varr.shape[0] > npart:
        part_idx = np.array_split(
            np.arange(varr.shape[0]), np.ceil(varr.shape[0] / npart)
        )
        tmp_ls = []
        sh_ls = []
        for idx in part_idx:
            cur_tmp, cur_motions = est_motion_chunk(
                varr[idx],
                [sh_org[i] for i in idx] if sh_org is not None else None,
                npart=npart,
                alt_error=alt_error,
                aggregation=aggregation,
                upsample=upsample,
                max_sh=max_sh,
                circ_thres=circ_thres,
                mesh_size=mesh_size,
                niter=niter,
                bin_thres=bin_thres,
            )
            tmp_ls.append(cur_tmp)
            sh_ls.append(cur_motions)
        varr = np.stack(tmp_ls, axis=0)
        sh_org = sh_ls
    # varr could have 4 dimensions in which case the second dimension has length
    # 3 representing the first, aggregated and the last frame of a chunk
    mask = np.ones_like(varr, dtype=bool)
    if bin_thres is not None and varr.ndim <= 3:
        for i, fm in enumerate(varr):
            mask[i] = fm > bin_thres
    good_fm = np.ones(varr.shape[0], dtype=bool)
    if circ_thres is not None and varr.ndim <= 3:
        for i, fm in enumerate(varr):
            good_fm[i] = check_temp(fm, max_sh) > circ_thres
    good_idxs = np.where(good_fm)[0].astype(int)
    prop_good = len(good_idxs) / len(good_fm)
    if prop_good < 0.9:
        warnings.warn(
            "only {} of the frames are good."
            "Consider lowering your circularity threshold".format(prop_good)
        )
    # use good frame closest to center as template
    mid = good_idxs[np.abs(good_idxs - varr.shape[0] / 2).argmin()]
    if mesh_size is not None:
        fm0 = varr[0, 0] if varr.ndim > 3 else varr[0]
        param = get_bspline_param(fm0, mesh_size)
        motions = np.zeros((varr.shape[0], 2, int(param[1]), int(param[0])))
    else:
        motions = np.zeros((varr.shape[0], 2))
    for i, fm in enumerate(varr):
        if i < mid:
            if varr.ndim > 3:
                src, dst = varr[i][1], varr[i + 1][1]
                src_ma, dst_ma = mask[i][1], mask[i + 1][1]
                if alt_error:
                    src_alt, dst_alt = varr[i][2], varr[i + 1][0]
                    src_alt_ma, dst_alt_ma = mask[i][2], mask[i + 1][0]
            else:
                # select the next good frame as template
                didx = good_idxs[good_idxs - (i + 1) >= 0][0]
                src, dst = varr[i], varr[didx]
                src_ma, dst_ma = mask[i], mask[didx]
            slc = slice(0, i + 1)
        elif i > mid:
            if varr.ndim > 3:
                src, dst = varr[i][1], varr[i - 1][1]
                src_ma, dst_ma = mask[i][1], mask[i - 1][1]
                if alt_error:
                    src_alt, dst_alt = varr[i][0], varr[i - 1][2]
                    src_alt_ma, dst_alt_ma = mask[i][0], mask[i - 1][2]
            else:
                # select the previous good frame as template
                didx = good_idxs[good_idxs - (i - 1) <= 0][-1]
                src, dst = varr[i], varr[didx]
                src_ma, dst_ma = mask[i], mask[didx]
            slc = slice(i, None)
        else:
            continue
        mo = est_motion_perframe(src, dst, upsample, src_ma, dst_ma, mesh_size, niter)
        if alt_error and varr.ndim > 3:
            mo_alt = est_motion_perframe(
                src_alt, dst_alt, upsample, src_alt_ma, dst_alt_ma, mesh_size, niter
            )
            if ((np.abs(mo - mo_alt) > alt_error).any()) and (
                np.abs(mo).sum() > np.abs(mo_alt).sum()
            ):
                mo = mo_alt
        # only add to the rest if current frame is good
    if good_fm[i]:
        # Ensure that the shape of 'mo' matches the shape of 'motions[i]'
        mo = np.array(mo)  # Convert 'mo' to a numpy array to ensure its shape
        if mo.shape != motions[i].shape:
            raise ValueError(
                f"Shape mismatch: Expected {motions[i].shape}, but got {mo.shape}"
            )
        motions[slc] = motions[slc] + mo
    else:
        # Handle cases where the current frame is not good
        # Ensure that the shape of 'mo' matches the shape of 'motions[i]'
        mo = np.array(mo)  # Convert 'mo' to a numpy array to ensure its shape
        if mo.shape != motions[i].shape:
            raise ValueError(
                f"Shape mismatch: Expected {motions[i].shape}, but got {mo.shape}"
            )
        motions[i] = motions[i] + mo
        # center shifts
    if mesh_size is not None:
        motions -= motions.mean(axis=(0, 2, 3), keepdims=True)
    else:
        motions -= motions.mean(axis=0)
    for i, v in enumerate(varr):
        if i not in good_idxs:
            continue
        if v.ndim > 2:
            for j, fm in enumerate(v):
                varr[i][j] = transform_perframe(fm, motions[i], fill=0)
        else:
            varr[i] = transform_perframe(v, motions[i], fill=0)
    varr = varr[good_idxs]
    if aggregation == "max":
        if varr.ndim > 3:
            tmp = varr.max(axis=(0, 1))
        else:
            tmp = varr.max(axis=0)
    elif aggregation == "mean":
        if varr.ndim > 3:
            tmp = varr.mean(axis=(0, 1))
        else:
            tmp = varr.mean(axis=0)
    else:
        raise ValueError("does not understand aggregation: {}".format(aggregation))
    if alt_error:
        if varr.ndim > 3:
            tmp0 = varr[0][0]
            tmp1 = varr[-1][-1]
        else:
            tmp0 = varr[0]
            tmp1 = varr[1]
        tmp = np.stack([tmp0, tmp, tmp1], axis=0)
    if sh_org is not None:
        motions = np.concatenate(
            [motions[i] + sh for i, sh in enumerate(sh_org)], axis=0
        )
    return tmp, motions

def check_temp(fm: np.ndarray, max_sh: int) -> float:
    """
    Compute the circularity metric for a frame.

    Parameters
    ----------
    fm : np.ndarray
        Input frame.
    max_sh : int
        Amount of zero padding when computing the comparison image.

    Returns
    -------
    circularity : float
        The circularity metric, will be `0` if the comparison image has more
        than one region with values `< 1`.

    See Also
    --------
    estimate_motion
    """
    fm_pad = np.pad(fm, max_sh)
    cor = cv2.matchTemplate(
        fm.astype(np.float32), fm_pad.astype(np.float32), cv2.TM_SQDIFF_NORMED
    )
    conts = cv2.findContours(
        (cor < 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    if len(conts) != 1:
        return 0
    cont = conts[0]
    perimeter = cv2.arcLength(cont, True)
    if perimeter <= 0:
        return 0
    area = cv2.contourArea(cont)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return circularity

def custom_arr_optimize(
    dsk: dict,
    keys: list,
    fast_funcs: list = FAST_FUNCTIONS,
    inline_patterns=[],
    rename_dict: Optional[dict] = None,
    rewrite_dict: Optional[dict] = None,
    keep_patterns=[],
) -> dict:
    """
    Customized implementation of array optimization function.

    Parameters
    ----------
    dsk : dict
        Input dask task graph.
    keys : list
        Output task keys.
    fast_funcs : list, optional
        List of fast functions to be inlined. By default :const:`FAST_FUNCTIONS`.
    inline_patterns : list, optional
        List of patterns of task keys to be inlined. By default `[]`.
    rename_dict : dict, optional
        Dictionary mapping old task keys to new ones. Only used during fusing of
        tasks. By default `None`.
    rewrite_dict : dict, optional
        Dictionary mapping old task key substrings to new ones. Applied at the
        end of optimization to all task keys. By default `None`.
    keep_patterns : list, optional
        List of patterns of task keys that should be preserved during
        optimization. By default `[]`.

    Returns
    -------
    dsk : dict
        Optimized dask graph.

    See Also
    -------
    :doc:`dask:optimize`
    `dask.array.optimization.optimize`
    """
    # inlining lots of array operations ref:
    # https://github.com/dask/dask/issues/6668
    if rename_dict:
        key_renamer = fct.partial(custom_fused_keys_renamer, rename_dict=rename_dict)
    else:
        key_renamer = custom_fused_keys_renamer
    keep_keys = []
    if keep_patterns:
        key_ls = list(dsk.keys())
        for pat in keep_patterns:
            keep_keys.extend(list(filter(lambda k: check_key(k, pat), key_ls)))
    dsk = darr.optimization.optimize(
        dsk,
        keys,
        fuse_keys=keep_keys,
        fast_functions=fast_funcs,
        rename_fused_keys=key_renamer,
    )
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if rewrite_dict:
        dsk_old = dsk.copy()
        for key, val in dsk_old.items():
            key_new = rewrite_key(key, rewrite_dict)
            if key_new != key:
                dsk[key_new] = val
                dsk[key] = key_new
    return dsk


def custom_arr_optimize(
    dsk: dict,
    keys: list,
    fast_funcs: list = FAST_FUNCTIONS,
    inline_patterns=[],
    rename_dict: Optional[dict] = None,
    rewrite_dict: Optional[dict] = None,
    keep_patterns=[],
) -> dict:
    """
    Customized implementation of array optimization function.

    Parameters
    ----------
    dsk : dict
        Input dask task graph.
    keys : list
        Output task keys.
    fast_funcs : list, optional
        List of fast functions to be inlined. By default :const:`FAST_FUNCTIONS`.
    inline_patterns : list, optional
        List of patterns of task keys to be inlined. By default `[]`.
    rename_dict : dict, optional
        Dictionary mapping old task keys to new ones. Only used during fusing of
        tasks. By default `None`.
    rewrite_dict : dict, optional
        Dictionary mapping old task key substrings to new ones. Applied at the
        end of optimization to all task keys. By default `None`.
    keep_patterns : list, optional
        List of patterns of task keys that should be preserved during
        optimization. By default `[]`.

    Returns
    -------
    dsk : dict
        Optimized dask graph.

    See Also
    -------
    :doc:`dask:optimize`
    `dask.array.optimization.optimize`
    """
    # inlining lots of array operations ref:
    # https://github.com/dask/dask/issues/6668
    if rename_dict:
        key_renamer = fct.partial(custom_fused_keys_renamer, rename_dict=rename_dict)
    else:
        key_renamer = custom_fused_keys_renamer
    keep_keys = []
    if keep_patterns:
        key_ls = list(dsk.keys())
        for pat in keep_patterns:
            keep_keys.extend(list(filter(lambda k: check_key(k, pat), key_ls)))
    dsk = darr.optimization.optimize(
        dsk,
        keys,
        fuse_keys=keep_keys,
        fast_functions=fast_funcs,
        rename_fused_keys=key_renamer,
    )
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if rewrite_dict:
        dsk_old = dsk.copy()
        for key, val in dsk_old.items():
            key_new = rewrite_key(key, rewrite_dict)
            if key_new != key:
                dsk[key_new] = val
                dsk[key] = key_new
    return dsk

def rewrite_key(key: Union[str, tuple], rwdict: dict) -> str:
    """
    Rewrite a task key according to `rwdict`.

    Parameters
    ----------
    key : Union[str, tuple]
        Input task key.
    rwdict : dict
        Dictionary mapping old task key substring to new ones. All keys in this
        dictionary that exists in input `key` will be substituted.

    Returns
    -------
    key : str
        The new key.

    Raises
    ------
    ValueError
        if input `key` is neither `str` or `tuple`
    """
    typ = type(key)
    if typ is tuple:
        k = key[0]
    elif typ is str:
        k = key
    else:
        raise ValueError("key must be either str or tuple: {}".format(key))
    for pat, repl in rwdict.items():
        k = re.sub(pat, repl, k)
    if typ is tuple:
        ret_key = list(key)
        ret_key[0] = k
        return tuple(ret_key)
    else:
        return k
    
def check_key(key: Union[str, tuple], pat: str) -> bool:
    """
    Check whether `key` contains pattern.

    Parameters
    ----------
    key : Union[str, tuple]
        Input key. If a `tuple` then the first element will be used to check.
    pat : str
        Pattern to check.

    Returns
    -------
    bool
        Whether `key` contains pattern.
    """
    try:
        return bool(re.search(pat, key))
    except TypeError:
        return bool(re.search(pat, key[0]))
    
def check_pat(key: Union[str, tuple], pat_ls: List[str]) -> bool:
    """
    Check whether `key` contains any pattern in a list.

    Parameters
    ----------
    key : Union[str, tuple]
        Input key. If a `tuple` then the first element will be used to check.
    pat_ls : List[str]
        List of pattern to check.

    Returns
    -------
    bool
        Whether `key` contains any pattern in the list.
    """
    for pat in pat_ls:
        if check_key(key, pat):
            return True
    return False


def inline_pattern(dsk: dict, pat_ls: List[str], inline_constants: bool) -> dict:
    """
    Inline tasks whose keys match certain patterns.

    Parameters
    ----------
    dsk : dict
        Input dask graph.
    pat_ls : List[str]
        List of patterns to check.
    inline_constants : bool
        Whether to inline constants.

    Returns
    -------
    dsk : dict
        Dask graph with keys inlined.

    See Also
    -------
    dask.optimization.inline
    """
    keys = [k for k in dsk.keys() if check_pat(k, pat_ls)]
    if keys:
        dsk = inline(dsk, keys, inline_constants=inline_constants)
        for k in keys:
            del dsk[k]
        if inline_constants:
            dsk, dep = cull(dsk, set(list(flatten(keys))))
    return dsk


def transform_perframe(
    fm: np.ndarray,
    tx_coef: np.ndarray,
    fill=0,
    param: Optional[np.ndarray] = None,
    mesh_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Transform a single frame.

    Parameters
    ----------
    fm : np.ndarray
        Frame to be transformed.
    tx_coef : np.ndarray
        Coefficient of transformation. If `tx_coef.ndim > 1`, then it is
        interpreted as BSpline transform coefficients. Otherwise it is
        interpreted as rigid translations.
    fill : int, optional
        Values used to fill in missing pixels (outside field of view). By
        default `0`.
    param : np.ndarray, optional
        Fixed parameters defining the BSpline transform. Only used if
        `tx_coef.ndim > 1`. By default `None`.
    mesh_size : Tuple[int, int], optional
        `mesh_size` parameter used to estimate motion. If `None` and
        `tx_coef.ndim > 1`, then one will be computed using
        :func:`get_mesh_size`. By default `None`.

    Returns
    -------
    fm : np.ndarray
        The frame after transform.
    """
    if tx_coef.ndim > 1:
        if param is None:
            if mesh_size is None:
                mesh_size = get_mesh_size(fm)
            param = get_bspline_param(fm, mesh_size)
        tx = sitk.BSplineTransform([sitk.GetImageFromArray(a) for a in tx_coef])
        tx.SetFixedParameters(param)
    else:
        tx = sitk.TranslationTransform(2, -tx_coef[::-1])
    fm = sitk.GetImageFromArray(fm)
    fm = sitk.Resample(fm, fm, tx, sitk.sitkLinear, fill)
    return sitk.GetArrayFromImage(fm)


def get_bspline_param(img: np.ndarray, mesh_size: Tuple[int, int]) -> np.ndarray:
    """
    Compute fixed parameters for the BSpline transform given a frame and mesh size.

    Parameters
    ----------
    img : np.ndarray
        Input frame.
    mesh_size : Tuple[int, int]
        Number of control points for the BSpline mesh.

    Returns
    -------
    param : np.ndarray
        Fixed parameters of a BSpline transform.
    """
    return sitk.BSplineTransformInitializer(
        image1=sitk.GetImageFromArray(img), transformDomainMeshSize=mesh_size
    ).GetFixedParameters()


def get_mesh_size(fm: np.ndarray) -> np.ndarray:
    """
    Compute suitable mesh size given a frame.

    The computed mesh size will result in approximately 100 pixels per
    patch/control point in each direction.

    Parameters
    ----------
    fm : np.ndarray
        The input frame.

    Returns
    -------
    mesh_size : np.ndarray
        The auto determined mesh size.
    """
    return (int(np.around(fm.shape[0] / 100)), int(np.around(fm.shape[1] / 100)))

def apply_transform(
    varr: xr.DataArray, trans: xr.DataArray, fill=0, mesh_size: Tuple[int, int] = None
) -> xr.DataArray:
    """
    Apply necessary transform to correct for motion.

    This function can correct for both rigid and non-rigid motion depending on
    the number of dimensions of input `trans`.

    Parameters
    ----------
    varr : xr.DataArray
        Input array representing movie data.
    trans : xr.DataArray
        Estimated motion, if `trans.ndim > 2` then it is interpreted as shifts
        of control points of mesh grid, and BSpline transform will be
        constructed. Otherwise it is interpreted as shifts in each direction of
        rigid translation.
    fill : int, optional
        Values used to fill in missing pixels (outside field of view). By default
        `0`.
    mesh_size : Tuple[int, int], optional
        `mesh_size` parameter used when estimating motion. Only used if
        `trans.ndim > 2`. If `None` and `trans.ndim > 2` then one will be
        computed using :func:`get_mesh_size`. By default `None`.

    Returns
    -------
    varr_sh : xr.DataArray
        Movie data after transform.
    """
    sh_dim = trans.coords["shift_dim"].values.tolist()
    if "grid0" in trans.dims:
        fm0 = varr.isel(frame=0).values
        if mesh_size is None:
            mesh_size = get_mesh_size(fm0)
        param = get_bspline_param(fm0, mesh_size)
        mdim = ["shift_dim", "grid0", "grid1"]
    else:
        param = None
        mdim = ["shift_dim"]
    varr_sh = xr.apply_ufunc(
        transform_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        trans,
        input_core_dims=[sh_dim, mdim],
        output_core_dims=[sh_dim],
        vectorize=True,
        dask="parallelized",
        kwargs={"fill": fill, "param": param},
        output_dtypes=[varr.dtype],
    )
    return varr_sh

def seeds_init(
    varr: xr.DataArray,
    wnd_size=500,
    method="rolling",
    stp_size=200,
    nchunk=100,
    max_wnd=10,
    diff_thres=2,
):
    """
    Generate over-complete set of seeds by finding local maxima across frames.

    This function computes the maximum intensity projection of a subset of
    frames and finds the local maxima. The subsetting use either a rolling
    window or random sampling of frames. `wnd_size` `stp_size` and `nchunk`
    controls different aspects of the subsetting. `max_wnd` and `diff_thres`
    controls how local maxima are computed. The set of all local maxima found in
    this process constitutes  an overly-complete set of seeds, representing
    putative locations of cells.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions "frame", "height" and "width".
    wnd_size : int, optional
        Number of frames in each chunk, for which a max projection will be
        calculated. By default `500`.
    method : str, optional
        Either `"rolling"` or `"random"`. Controls whether to use rolling window
        or random sampling of frames to construct chunks. By default
        `"rolling"`.
    stp_size : int, optional
        Number of frames between the center of each chunk when stepping through
        the data with rolling windows. Only used if `method is "rolling"`. By
        default `200`.
    nchunk : int, optional
        Number of chunks to sample randomly. Only used if `method is "random"`.
        By default `100`.
    max_wnd : int, optional
        Radius (in pixels) of the disk window used for computing local maxima.
        Local maximas are defined as pixels with maximum intensity in such a
        window. By default `10`.
    diff_thres : int, optional
        Intensity threshold for the difference between local maxima and its
        neighbours. Any local maxima that is not birghter than its neighbor
        (defined by the same disk window) by `diff_thres` intensity values will
        be filtered out. By default `2`.

    Returns
    -------
    seeds : pd.DataFrame
        Seeds dataframe with each seed as a row. Has column "height" and "width"
        which are location of the seeds. Also has column "seeds" which is an
        integer showing how many chunks where the seed is considered a local
        maxima.
    """
    print("constructing chunks")
    if hasattr(varr, "coords"):
        idx_fm = varr.coords["frame"]
        nfm = len(idx_fm)
    elif isinstance(varr, np.ndarray):
        nfm=1
    else:
        print('unrecognized input')
    if method == "rolling":
        nstp = np.ceil(nfm / stp_size) + 1
        centers = np.linspace(0, nfm - 1, int(nstp))
        hwnd = np.ceil(wnd_size / 2)
        max_idx = list(
            map(
                lambda c: slice(
                    int(np.floor(c - hwnd).clip(0)), int(np.ceil(c + hwnd))
                ),
                centers,
            )
        )
    elif method == "random":
        max_idx = [np.random.randint(0, nfm - 1, wnd_size) for _ in range(nchunk)]
    print("computing max projections")
    if hasattr(varr, "coords"):
        res = [max_proj_frame(varr, cur_idx) for cur_idx in max_idx]
        max_res = xr.concat(res, "sample")
        max_res = save_minian(max_res.rename("max_res"), dpath, overwrite=True)
    elif isinstance(varr, np.ndarray):
        max_res = varr
    print("calculating local maximum")
    if hasattr(varr, "coords"):
        loc_max = xr.apply_ufunc(
            local_max_roll,
            max_res,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.uint8],
            kwargs=dict(k0=2, k1=max_wnd, diff=diff_thres),
        ).sum("sample")
        seeds = (
            loc_max.where(loc_max > 0).rename("seeds").to_dataframe().dropna().reset_index()
        )
    else:
        if max_wnd <= 1:
            max_wnd = 100
        loc_max = local_max_roll(max_res, k0=1, k1=max_wnd, diff=diff_thres)
        if not isinstance(loc_max, xr.DataArray):  # Check if loc_max is already a DataArray
            loc_max = xr.DataArray(loc_max, dims=["height", "width"])
        seeds = (
            loc_max.where(loc_max > 0, drop=True)
            .rename("seeds")
            .to_dataframe()
            .dropna()
            .reset_index()
        )
    return seeds[["height", "width", "seeds"]]

def max_proj_frame(varr: xr.DataArray, idx: np.ndarray) -> xr.DataArray:
    """
    Compute max projection on a given subset of frames.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data containing all frames.
    idx : np.ndarray
        The subset of frames to use to compute max projection.

    Returns
    -------
    max_proj : xr.DataArray
        The max projection.
    """
    return varr.isel(frame=idx).max("frame")


def local_max_roll(
    fm: np.ndarray, k0: int, k1: int, diff: Union[int, float]
) -> np.ndarray:
    """
    Compute local maxima of a frame with a range of kernel size.

    This function wraps around :func:`minian.utilities.local_extreme` and
    compute local maxima of the input frame with kernels of size ranging from
    `k0` to `k1`. It then takes the union of all the local maxima, and
    additionally merge all the connecting local maxima by using the middle
    pixel.

    Parameters
    ----------
    fm : np.ndarray
        The input frame.
    k0 : int
        The lower bound (inclusive) of the range of kernel sizes.
    k1 : int
        The upper bound (inclusive) of the range of kernel sizes.
    diff : Union[int, float]
        Intensity threshold for the difference between local maxima and its
        neighbours, passed to :func:`minian.utilities.local_extreme`.

    Returns
    -------
    max_res : np.ndarray
        The image of local maxima. Has same shape as `fm`, and 1 at local
        maxima.
    """
    if k0 >= k1:
        raise ValueError("k0 must be less than k1")
    
    max_ls = []
    if k0 == k1:  # Only one kernel size
        selem = disk(k0)
        fm_max = local_extreme(fm, selem, diff=diff)
        max_ls.append(fm_max)
    else:
        for ksize in range(k0, k1):
            selem = disk(ksize)
            fm_max = local_extreme(fm, selem, diff=diff)
            max_ls.append(fm_max)
    lmax = (np.stack(max_ls, axis=0).sum(axis=0) > 0).astype(np.uint8)
    nlab, max_lab = cv2.connectedComponents(lmax)
    max_res = np.zeros_like(lmax)
    for lb in range(1, nlab):
        area = max_lab == lb
        if np.sum(area) > 1:
            crds = tuple(int(np.median(c)) for c in np.where(area))
            max_res[crds] = 1
        else:
            max_res[np.where(area)] = 1
    return max_res

def local_extreme(fm: np.ndarray, k: np.ndarray, etype="max", diff=0) -> np.ndarray:
    """
    Find local extreme of a 2d array.

    Parameters
    ----------
    fm : np.ndarray
        The input 2d array.
    k : np.ndarray
        Structuring element defining the locality of the result, passed as
        `kernel` to :func:`cv2.erode` and :func:`cv2.dilate`.
    etype : str, optional
        Type of local extreme. Either `"min"` or `"max"`. By default `"max"`.
    diff : int, optional
        Threshold of difference between local extreme and its neighbours. By
        default `0`.

    Returns
    -------
    fm_ext : np.ndarray
        The returned 2d array whose non-zero elements represent the location of
        local extremes.

    Raises
    ------
    ValueError
        if `etype` is not "min" or "max"
    """
    # Convert fm to uint8 if it's not already in that format
    if fm.dtype != np.uint8:
        fm = fm.astype(np.uint8)
        
    fm_max = cv2.dilate(fm, k)
    fm_min = cv2.erode(fm, k)
    fm_diff = ((fm_max - fm_min) > diff).astype(np.uint8)
    if etype == "max":
        fm_ext = (fm == fm_max).astype(np.uint8)
    elif etype == "min":
        fm_ext = (fm == fm_min).astype(np.uint8)
    else:
        raise ValueError("Don't understand {}".format(etype))
    return cv2.bitwise_and(fm_ext, fm_diff).astype(np.uint8)



def gmm_refine(
    varr: xr.DataArray,
    seeds: pd.DataFrame,
    q=(0.1, 99.9),
    n_components=2,
    valid_components=1,
    mean_mask=True,
) -> Tuple[pd.DataFrame, xr.DataArray, GaussianMixture]:
    """
    Filter seeds by fitting a GMM to peak-to-peak values.

    This function assume that the distribution of peak-to-peak values of
    fluorescence across all seeds can be model by a Gaussian Mixture Model (GMM)
    with different means. It computes peak-to-peak value for all the seeds, then
    fit a GMM with `n_components` to the distribution, and filter out the seeds
    belonging to the `n_components - valid_components` number of gaussians with
    lower means.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data. Should have dimension "spatial" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    q : tuple, optional
        Percentile to use to compute the peak-to-peak values. For a given seed
        with corresponding fluorescent fluctuation `f`, the peak-to-peak value
        for that seed is computed as `np.percentile(f, q[1]) - np.percentile(f,
        q[0])`. By default `(0.1, 99.9)`.
    n_components : int, optional
        Number of components (Gaussians) in the GMM model. By default `2`.
    valid_components : int, optional
        Number of components (Gaussians) to be considered as modeling the
        distribution of peak-to-peak values of valid seeds. Should be smaller
        than `n_components`. By default `1`.
    mean_mask : bool, optional
        Whether to apply additional criteria where a seed is valid only if its
        peak-to-peak value exceeds the mean of the lowest gaussian distribution.
        Only useful in corner cases where the distribution of the gaussian
        heavily overlap. By default `True`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_gmm",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds` it will be overwritten.
    varr_pv : xr.DataArray
        The computed peak-to-peak values for each seeds.
    gmm : GaussianMixture
        The fitted GMM model object.

    See Also
    -------
    sklearn.mixture.GaussianMixture
    """
    print("selecting seeds")
    varr_sub = varr.sel(spatial=[tuple(hw) for hw in seeds[["height", "width"]].values])
    print("computing peak-valley values")
    varr_valley = xr.apply_ufunc(
        np.percentile,
        varr_sub.chunk(dict(frame=-1)),
        input_core_dims=[["frame"]],
        kwargs=dict(q=q[0], axis=-1),
        dask="parallelized",
        output_dtypes=[varr_sub.dtype],
    )
    varr_peak = xr.apply_ufunc(
        np.percentile,
        varr_sub.chunk(dict(frame=-1)),
        input_core_dims=[["frame"]],
        kwargs=dict(q=q[1], axis=-1),
        dask="parallelized",
        output_dtypes=[varr_sub.dtype],
    )
    varr_pv = varr_peak - varr_valley
    varr_pv = varr_pv.compute()
    print("fitting GMM models")
    dat = varr_pv.values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(dat)
    idg = np.argsort(gmm.means_.reshape(-1))[-valid_components:]
    idx_valid = np.isin(gmm.predict(dat), idg)
    if mean_mask:
        idx_mean = dat > np.sort(gmm.means_)[0]
        idx_valid = np.logical_and(idx_mean.squeeze(), idx_valid)
    seeds["mask_gmm"] = idx_valid
    return seeds, varr_pv, gmm


def pnr_refine(
    varr: xr.DataArray,
    seeds: pd.DataFrame,
    noise_freq=0.25,
    thres: Union[float, str] = 1.5,
    q=(0.1, 99.9),
    med_wnd: Optional[int] = None,
) -> Tuple[pd.DataFrame, xr.DataArray, Optional[GaussianMixture]]:
    """
    Filter seeds by thresholding peak-to-noise ratio.

    For each input seed, the noise is defined as high-pass filtered fluorescence
    trace of the seed. The peak-to-noise ratio (pnr) of that seed is then
    defined as the ratio between the peak-to-peak value of the originial
    fluorescence trace and that of the noise trace. Optionally, if abrupt
    changes in baseline fluorescence is expected, then the baseline can be
    estimated by median-filtering the fluorescence trace and subtracted from the
    original trace before computing the peak-to-noise ratio. In addition, if a
    hard threshold of pnr is not desired, then a Gaussian Mixture Model with 2
    components can be fitted to the distribution of pnr across all seeds, and
    only seeds with pnr belonging to the higher-mean Gaussian will be considered
    valide.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data, should have dimensions "height", "width" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    noise_freq : float, optional
        Cut-off frequency for the high-pass filter used to define noise,
        specified as fraction of sampling frequency. By default `0.25`.
    thres : Union[float, str], optional
        Threshold of the peak-to-noise ratio. If `"auto"` then a :class:`GMM
        <sklearn.mixture.GaussianMixture>` will be fit to the distribution of
        pnr. By default `1.5`.
    q : tuple, optional
        Percentile to use to compute the peak-to-peak values. For a given
        fluorescence fluctuation `f`, the peak-to-peak value for that seed is
        computed as `np.percentile(f, q[1]) - np.percentile(f, q[0])`. By
        default `(0.1, 99.9)`.
    med_wnd : int, optional
        Size of the median filter window to remove baseline. If `None` then no
        filtering will be done. By default `None`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_pnr",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds` it will be overwritten.
    pnr : xr.DataArray
        The computed peak-to-noise ratio for each seeds.
    gmm : GaussianMixture, optional
        The GMM model object fitted to the distribution of pnr. Will be `None`
        unless `thres` is `"auto"`.
    """
    print("selecting seeds")
    # vectorized indexing on dask arrays produce a single chunk.
    # to memory issue, split seeds into 128 chunks, with chunk size no greater than 100
    chk_size = min(int(len(seeds) / 128), 100)
    vsub_ls = []
    for _, seed_sub in seeds.groupby(np.arange(len(seeds)) // chk_size):
        vsub = varr.sel(
            height=seed_sub["height"].to_xarray(), width=seed_sub["width"].to_xarray()
        )
        vsub_ls.append(vsub)
    varr_sub = xr.concat(vsub_ls, "index")
    if med_wnd:
        print("removing baseline")
        varr = xr.apply_ufunc(
            med_baseline,
            varr_sub,
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            dask="parallelized",
            kwargs={"wnd": med_wnd},
            vectorize=True,
            output_dtypes=[varr.dtype],
        )
    print("computing peak-noise ratio")
    pnr = xr.apply_ufunc(
        pnr_perseed,
        varr_sub,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        kwargs={"freq": noise_freq, "q": q},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).compute()
    if thres == "auto":
        gmm = GaussianMixture(n_components=2)
        gmm.fit(np.nan_to_num(pnr.values.reshape(-1, 1)))
        idg = np.argsort(gmm.means_.reshape(-1))[-1]
        idx_valid = np.isin(gmm.predict(pnr.values.reshape(-1, 1)), idg)
        seeds["mask_pnr"] = idx_valid
    else:
        mask = pnr > thres
        mask_df = mask.to_pandas().rename("mask_pnr")
        seeds["mask_pnr"] = mask_df
        gmm = None
    return seeds, pnr, gmm

def ptp_q(a: np.ndarray, q: tuple) -> float:
    """
    Compute peak-to-peak value of input with percentile values.

    Parameters
    ----------
    a : np.ndarray
        Input array.
    q : tuple
        Tuple specifying low and high percentile values.

    Returns
    -------
    ptp : float
        The peak-to-peak value.
    """
    return np.percentile(a, q[1]) - np.percentile(a, q[0])


def pnr_perseed(a: np.ndarray, freq: float, q: tuple) -> float:
    """
    Compute peak-to-noise ratio of a given timeseries.

    Parameters
    ----------
    a : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency of the high-pass filtering used to define noise.
    q : tuple
        Percentile used to compute peak-to-peak values.

    Returns
    -------
    pnr : float
        Peak-to-noise ratio.

    See Also
    -------
    pnr_refine : for definition of peak-to-noise ratio
    """
    ptp = ptp_q(a, q)
    a = filt_fft(a, freq, btype="high")
    ptp_noise = ptp_q(a, q)
    
    if ptp_noise == 0:
        # Handle division by zero gracefully
        return float('inf')  # Return positive infinity
        
    return ptp / ptp_noise

def intensity_refine(
    varr: xr.DataArray, seeds: pd.DataFrame, thres_mul=2
) -> pd.DataFrame:
    """
    Filter seeds by thresholding the intensity of their corresponding pixels in
    the max projection of the movie.

    This function generate a histogram of the max projection by spliting the
    intensity into bins of roughly 10 pixels. Then the intensity threshold is
    defined as the intensity of the peak of the histogram times `thres_mul`.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions "height", "width" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    thres_mul : int, optional
        Scalar multiplied to the intensity value corresponding to the peak of
        max projection histogram. By default `2`, which can be interpreted as
        "seeds are only valid if they are more than twice as bright as the
        majority of the pixels".

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_int",
        indicating whether the seed is considered valid by this function.
    """
    try:
        fm_max = varr.max("frame")
    except ValueError:
        print("using input as max projection")
        fm_max = varr
    bins = np.around(fm_max.sizes["height"] * fm_max.sizes["width"] / 10).astype(int)
    hist, edges = np.histogram(fm_max, bins=bins)
    try:
        thres = edges[int(np.around(np.argmax(hist) * thres_mul))]
    except IndexError:
        print("threshold out of bound, returning input")
        return seeds
    mask = (fm_max > thres).stack(spatial=["height", "width"])
    mask_df = mask.to_pandas().rename("mask_int").reset_index()
    seeds = pd.merge(seeds, mask_df, on=["height", "width"], how="left")
    return seeds


def ks_refine(varr: xr.DataArray, seeds: pd.DataFrame, sig=0.01) -> pd.DataFrame:
    """
    Filter the seeds using Kolmogorov-Smirnov (KS) test.

    This function assume that the valid seeds’ fluorescence across frames
    notionally follows a bimodal distribution: with a large normal distribution
    representing baseline activity, and a second peak representing when the
    seed/cell is active. KS allows to discard the seeds where the
    null-hypothesis (i.e. the fluorescence intensity is simply a normal
    distribution) is rejected at `sig`.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions "height", "width" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    sig : float, optional
        The significance threshold to reject null-hypothesis. By default `0.01`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_ks",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds` it will be overwritten.
    """
    print("selecting seeds")
    # vectorized indexing on dask arrays produce a single chunk.
    # to memory issue, split seeds into 128 chunks, with chunk size no greater than 100
    chk_size = min(int(len(seeds) / 128), 100)
    vsub_ls = []
    for _, seed_sub in seeds.groupby(np.arange(len(seeds)) // chk_size):
        vsub = varr.sel(
            height=seed_sub["height"].to_xarray(), width=seed_sub["width"].to_xarray()
        )
        vsub_ls.append(vsub)
    varr_sub = xr.concat(vsub_ls, "index")
    print("performing KS test")
    ks = xr.apply_ufunc(
        ks_perseed,
        varr_sub,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).compute()
    ks = (ks < sig).to_pandas().rename("mask_ks")
    seeds["mask_ks"] = ks
    return seeds



def ks_perseed(a: np.ndarray) -> float:
    """
    Perform KS test on input and return the p-value.

    Parameters
    ----------
    a : np.ndarray
        Input data.

    Returns
    -------
    p : float
        The p-value of the KS test.

    See Also
    -------
    scipy.stats.kstest
    """
    a = zscore(a)
    return kstest(a, "norm")[1]


def seeds_merge(
    varr: xr.DataArray,
    max_proj: xr.DataArray,
    seeds: pd.DataFrame,
    thres_dist=5,
    thres_corr=0.6,
    noise_freq: Optional[float] = None,
) -> pd.DataFrame:
    """
    Merge seeds based on spatial distance and temporal correlation of their
    activities.

    This function build an adjacency matrix by thresholding spatial distance
    between seeds and temporal correlation between activities of seeds. It then
    merge seeds using the adjacency matrix by only keeping the seed with maximum
    intensity in the max projection within each connected group of seeds. The
    merge is therefore transitive.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimension "height", "width" and "frame".
    max_proj : xr.DataArray
        Max projection of the movie data.
    seeds : pd.DataFrame
        Dataframe of seeds to be merged.
    thres_dist : int, optional
        Threshold of distance between seeds in pixel. By default `5`.
    thres_corr : float, optional
        Threshold of temporal correlation between activities of seeds. By
        default `0.6`.
    noise_freq : float, optional
        Cut-off frequency for optional smoothing of activities before computing
        the correlation. If `None` then no smoothing will be done. By default
        `None`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_mrg",
        indicating whether the seed should be kept after the merge. If the
        column already exists in input `seeds` it will be overwritten.
    """
    print("computing distance")
    nng = radius_neighbors_graph(seeds[["height", "width"]], thres_dist)
    print("computing correlations")
    adj = adj_corr(varr, nng, seeds[["height", "width"]], noise_freq)
    print("merging seeds")
    adj = adj > thres_corr
    adj = adj + adj.T
    labels = label_connected(adj, only_connected=True)
    iso = np.where(labels < 0)[0]
    seeds_final = set(iso.tolist())
    for cur_cmp in np.unique(labels):
        if cur_cmp < 0:
            continue
        cur_smp = np.where(labels == cur_cmp)[0]
        cur_max = np.array(
            [
                max_proj.sel(
                    height=seeds.iloc[s]["height"], width=seeds.iloc[s]["width"]
                )
                for s in cur_smp
            ]
        )
        max_seed = cur_smp[np.argmax(cur_max)]
        seeds_final.add(max_seed)
    seeds["mask_mrg"] = False
    seeds.loc[list(seeds_final), "mask_mrg"] = True
    return seeds


def initA(
    varr: xr.DataArray,
    seeds: pd.DataFrame,
    thres_corr=0.8,
    wnd=10,
    noise_freq: Optional[float] = None,
) -> xr.DataArray:
    """
    Initialize spatial footprints from seeds.

    For each input seed, this function compute the correlation between the
    fluorescence activity of the seed and those of its neighboring pixels up to
    `wnd` pixels. It then set all correlation below `thres_corr` to zero, and
    use the resulting correlation image as the resutling spatial footprint of
    the seed.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimension "height", "width" and "frame".
    seeds : pd.DataFrame
        Dataframe of seeds.
    thres_corr : float, optional
        Threshold of correlation, below which the values will be set to zero in
        the resulting spatial footprints. By default `0.8`.
    wnd : int, optional
        Radius (in pixels) of a disk window within which correlation will be
        computed for each seed. By default `10`.
    noise_freq : float, optional
        Cut-off frequency for optional smoothing of activities before computing
        the correlation. If `None` then no smoothing will be done. By default
        `None`.

    Returns
    -------
    A : xr.DataArray
        The initial estimation of spatial footprint for each cell. Should have
        dimensions ("unit_id", "height", "width").

    See Also
    -------
    minian.cnmf.graph_optimize_corr :
        for how the correlation are computed in an out-of-core fashion
    """
    print("optimizing computation graph")
    nod_df = pd.DataFrame(
        np.array(
            list(itt.product(varr.coords["height"].values, varr.coords["width"].values))
        ),
        columns=["height", "width"],
    ).merge(seeds.reset_index(), how="outer", on=["height", "width"])
    seed_df = nod_df[nod_df["index"].notnull()]
    nn_tree = KDTree(nod_df[["height", "width"]], leaf_size=(2 * wnd) ** 2)
    nns_arr = nn_tree.query_radius(seed_df[["height", "width"]], r=wnd)
    sdg = nx.Graph()
    sdg.add_nodes_from(
        [
            (i, d)
            for i, d in enumerate(
                nod_df[["height", "width", "index"]].to_dict("records")
            )
        ]
    )
    for isd, nns in enumerate(nns_arr):
        cur_sd = seed_df.index[isd]
        sdg.add_edges_from([(cur_sd, n) for n in nns if n != cur_sd])
    sdg.remove_nodes_from(list(nx.isolates(sdg)))
    sdg = nx.convert_node_labels_to_integers(sdg)
    corr_df = graph_optimize_corr(varr, sdg, noise_freq)
    print("building spatial matrix")
    corr_df = corr_df[corr_df["corr"] > thres_corr]
    nod_df = pd.DataFrame.from_dict(dict(sdg.nodes(data=True)), orient="index")
    seed_df = nod_df[nod_df["index"].notnull()].astype({"index": int})
    A_ls = []
    ih_dict = (
        varr.coords["height"]
        .to_series()
        .reset_index(drop=True)
        .reset_index()
        .set_index("height")["index"]
        .to_dict()
    )
    iw_dict = (
        varr.coords["width"]
        .to_series()
        .reset_index(drop=True)
        .reset_index()
        .set_index("width")["index"]
        .to_dict()
    )
    Ashape = (varr.sizes["height"], varr.sizes["width"])
    for seed_id, sd in seed_df.iterrows():
        src_corr = corr_df[corr_df["target"] == seed_id].copy()
        src_nods = nod_df.loc[src_corr["source"]]
        src_corr["height"], src_corr["width"] = (
            src_nods["height"].values,
            src_nods["width"].values,
        )
        tgt_corr = corr_df[corr_df["source"] == seed_id].copy()
        tgt_nods = nod_df.loc[tgt_corr["target"]]
        tgt_corr["height"], tgt_corr["width"] = (
            tgt_nods["height"].values,
            tgt_nods["width"].values,
        )
        cur_corr = pd.concat([src_corr, tgt_corr])._append(
            {"corr": 1, "height": sd["height"], "width": sd["width"]}, ignore_index=True
        )
        cur_corr["iheight"] = cur_corr["height"].map(ih_dict)
        cur_corr["iwidth"] = cur_corr["width"].map(iw_dict)
        cur_A = darr.array(
            sparse.COO(
                cur_corr[["iheight", "iwidth"]].T, cur_corr["corr"], shape=Ashape
            )
        )
        A_ls.append(cur_A)
    A = xr.DataArray(
        darr.stack(A_ls).map_blocks(lambda a: a.todense(), dtype=float),
        dims=["unit_id", "height", "width"],
        coords={
            "unit_id": seed_df["index"].values,
            "height": varr.coords["height"].values,
            "width": varr.coords["width"].values,
        },
    )
    return A


def initC(varr: xr.DataArray, A: xr.DataArray) -> xr.DataArray:
    """
    Initialize temporal component given spatial footprints.

    The temporal component is computed as the least-square solution between the
    input movie and the spatial footprints over the "height" and "width"
    dimensions.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions ("height", "width", "frame").
    A : xr.DataArray
        Spatial footprints of cells. Should have dimensions ("unit_id",
        "height", "width").

    Returns
    -------
    C : xr.DataArray
        The initial estimation of temporal components for each cell. Should have
        dimensions ("unit_id", "frame").
    """
    uids = A.coords["unit_id"]
    fms = varr.coords["frame"]
    A = (
        A.stack(spatial=["height", "width"])
        .transpose("spatial", "unit_id")
        .data.map_blocks(scipy.sparse.csc_matrix)
        .rechunk(-1)
        .persist()
    )
    varr = varr.stack(spatial=["height", "width"]).transpose("frame", "spatial").data
    C = sps_lstsq(A, varr, iter_lim=10)
    C = xr.DataArray(
        C, dims=["frame", "unit_id"], coords={"unit_id": uids, "frame": fms}
    ).transpose("unit_id", "frame")
    return C


@darr.as_gufunc(signature="(h, w)->(h, w)", output_dtypes=int, allow_rechunk=True)
def da_label(im: np.ndarray) -> np.ndarray:
    """
    Label connected features in a 2d array.

    Parameters
    ----------
    im : np.ndarray
        Input array.

    Returns
    -------
    label : np.ndarray
        Label array. Should have same shape as input `im`.

    See Also
    -------
    scipy.ndimage.label
    """
    return label(im)[0]

def load_videos(
    vpath: str,
    pattern=r"msCam[0-9]+\.avi$",
    dtype: Union[str, type] = np.float64,
    downsample: Optional[dict] = None,
    downsample_strategy="subset",
    post_process: Optional[Callable] = None,
) -> xr.DataArray:
    """
    Load multiple videos in a folder and return a `xr.DataArray`.

    Load videos from the folder specified in `vpath` and according to the regex
    `pattern`, then concatenate them together and return a `xr.DataArray`
    representation of the concatenated videos. The videos are sorted by
    filenames with :func:`natsort.natsorted` before concatenation. Optionally
    the data can be downsampled, and the user can pass in a custom callable to
    post-process the result.

    Parameters
    ----------
    vpath : str
        The path containing the videos to load.
    pattern : regexp, optional
        The regexp matching the filenames of the videso. By default
        `r"msCam[0-9]+\.avi$"`, which can be interpreted as filenames starting
        with "msCam" followed by at least a number, and then followed by ".avi".
    dtype : Union[str, type], optional
        Datatype of the resulting DataArray, by default `np.float64`.
    downsample : dict, optional
        A dictionary mapping dimension names to an integer downsampling factor.
        The dimension names should be one of "height", "width" or "frame". By
        default `None`.
    downsample_strategy : str, optional
        How the downsampling should be done. Only used if `downsample` is not
        `None`. Either `"subset"` where data points are taken at an interval
        specified in `downsample`, or `"mean"` where mean will be taken over
        data within each interval. By default `"subset"`.
    post_process : Callable, optional
        An user-supplied custom function to post-process the resulting array.
        Four arguments will be passed to the function: the resulting DataArray
        `varr`, the input path `vpath`, the list of matched video filenames
        `vlist`, and the list of DataArray before concatenation `varr_list`. The
        function should output another valide DataArray. In other words, the
        function should have signature `f(varr: xr.DataArray, vpath: str, vlist:
        List[str], varr_list: List[xr.DataArray]) -> xr.DataArray`. By default
        `None`

    Returns
    -------
    varr : xr.DataArray
        The resulting array representation of the input movie. Should have
        dimensions ("frame", "height", "width").

    Raises
    ------
    FileNotFoundError
        if no files under `vpath` match the pattern `pattern`
    ValueError
        if the matched files does not have extension ".avi", ".mkv" or ".tif"
    NotImplementedError
        if `downsample_strategy` is not "subset" or "mean"
    """
    vpath = os.path.normpath(vpath)
    vlist = natsorted(
        [vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)]
    )
    if not vlist:
        raise FileNotFoundError(
            "No data with pattern {}"
            " found in the specified folder {}".format(pattern, vpath)
        )
    print("loading {} videos in folder {}".format(len(vlist), vpath))

    file_extension = os.path.splitext(vlist[0])[1]
    if file_extension in (".avi", ".mkv"):
        movie_load_func = load_avi_lazy
    elif file_extension == ".tif":
        movie_load_func = load_tif_lazy
    else:
        raise ValueError("Extension not supported.")

    varr_list = [movie_load_func(v) for v in vlist]
    varr = darr.concatenate(varr_list, axis=0)
    varr = xr.DataArray(
        varr,
        dims=["frame", "height", "width"],
        coords=dict(
            frame=np.arange(varr.shape[0]),
            height=np.arange(varr.shape[1]),
            width=np.arange(varr.shape[2]),
        ),
    )
    if dtype:
        varr = varr.astype(dtype)
    if downsample:
        if downsample_strategy == "mean":
            varr = varr.coarsen(**downsample, boundary="trim", coord_func="min").mean()
        elif downsample_strategy == "subset":
            varr = varr.isel(**{d: slice(None, None, w) for d, w in downsample.items()})
        else:
            raise NotImplementedError("unrecognized downsampling strategy")
    varr = varr.rename("fluorescence")
    if post_process:
        varr = post_process(varr, vpath, vlist, varr_list)
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^load_avi_ffmpeg"])
    with da.config.set(array_optimize=arr_opt):
        varr = da.optimize(varr)[0]
    return varr


def load_tif_lazy(fname: str) -> darr.array:
    """
    Lazy load a tif stack of images.

    Parameters
    ----------
    fname : str
        The filename of the tif stack to load.

    Returns
    -------
    arr : darr.array
        Resulting dask array representation of the tif stack.
    """
    data = TiffFile(fname)
    f = len(data.pages)

    fmread = da.delayed(load_tif_perframe)
    flist = [fmread(fname, i) for i in range(f)]

    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_tif_perframe(fname: str, fid: int) -> np.ndarray:
    """
    Load a single image from a tif stack.

    Parameters
    ----------
    fname : str
        The filename of the tif stack.
    fid : int
        The index of the image to load.

    Returns
    -------
    arr : np.ndarray
        Array representation of the image.
    """
    return imread(fname, key=fid)


def load_avi_lazy_framewise(fname: str) -> darr.array:
    cap = cv2.VideoCapture(fname)
    f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fmread = da.delayed(load_avi_perframe)
    flist = [fmread(fname, i) for i in range(f)]
    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_avi_lazy(fname: str) -> darr.array:
    """
    Lazy load an avi video.

    This function construct a single delayed task for loading the video as a
    whole.

    Parameters
    ----------
    fname : str
        The filename of the video to load.

    Returns
    -------
    arr : darr.array
        The array representation of the video.
    """
    probe = ffmpeg.probe(fname)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w = int(video_info["width"])
    h = int(video_info["height"])
    f = int(video_info["nb_frames"])
    return da.array.from_delayed(
        da.delayed(load_avi_ffmpeg)(fname, h, w, f), dtype=np.uint8, shape=(f, h, w)
    )


def load_avi_ffmpeg(fname: str, h: int, w: int, f: int) -> np.ndarray:
    """
    Load an avi video using `ffmpeg`.

    This function directly invoke `ffmpeg` using the `python-ffmpeg` wrapper and
    retrieve the data from buffer.

    Parameters
    ----------
    fname : str
        The filename of the video to load.
    h : int
        The height of the video.
    w : int
        The width of the video.
    f : int
        The number of frames in the video.

    Returns
    -------
    arr : np.ndarray
        The resulting array. Has shape (`f`, `h`, `w`).
    """
    out_bytes, err = (
        ffmpeg.input(fname)
        .video.output("pipe:", format="rawvideo", pix_fmt="gray")
        .run(capture_stdout=True)
    )
    return np.frombuffer(out_bytes, np.uint8).reshape(f, h, w)


def load_avi_perframe(fname: str, fid: int) -> np.ndarray:
    cap = cv2.VideoCapture(fname)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, fm = cap.read()
    if ret:
        return np.flip(cv2.cvtColor(fm, cv2.COLOR_RGB2GRAY), axis=0)
    else:
        print("frame read failed for frame {}".format(fid))
        return np.zeros((h, w))


def open_minian(
     post_process: Optional[Callable] = None, return_dict=False
) -> Union[dict, xr.Dataset]:
    """
    Load an existing minian dataset.

    If `dpath` is a file, then it is assumed that the full dataset is saved as a
    single file, and this function will directly call
    :func:`xarray.open_dataset` on `dpath`. Otherwise if `dpath` is a directory,
    then it is assumed that the dataset is saved as a directory of `zarr`
    arrays, as produced by :func:`save_minian`. This function will then iterate
    through all the directories under input `dpath` and load them as
    `xr.DataArray` with `zarr` backend, so it is important that the user make
    sure every directory under `dpath` can be load this way. The loaded arrays
    will be combined as either a `xr.Dataset` or a `dict`. Optionally a
    user-supplied custom function can be used to post process the resulting
    `xr.Dataset`.

    Parameters
    ----------
    dpath : str
        The path to the minian dataset that should be loaded.
    post_process : Callable, optional
        User-supplied function to post process the dataset. Only used if
        `return_dict` is `False`. Two arguments will be passed to the function:
        the resulting dataset `ds` and the data path `dpath`. In other words the
        function should have signature `f(ds: xr.Dataset, dpath: str) ->
        xr.Dataset`. By default `None`.
    return_dict : bool, optional
        Whether to combine the DataArray as dictionary, where the `.name`
        attribute will be used as key. Otherwise the DataArray will be combined
        using `xr.merge(..., compat="no_conflicts")`, which will implicitly
        align the DataArray over all dimensions, so it is important to make sure
        the coordinates are compatible and will not result in creation of large
        NaN-padded results. Only used if `dpath` is a directory, otherwise a
        `xr.Dataset` is always returned. By default `False`.

    Returns
    -------
    ds : Union[dict, xr.Dataset]
        The resulting dataset. If `return_dict` is `True` it will be a `dict`,
        otherwise a `xr.Dataset`.

    See Also
    -------
    xarray.open_zarr : for how each directory will be loaded as `xr.DataArray`
    xarray.merge : for how the `xr.DataArray` will be merged as `xr.Dataset`
    """
    if isfile(dpath):
        ds = xr.open_dataset(dpath).chunk()
    elif isdir(dpath):
        dslist = []
        for d in listdir(dpath):
            arr_path = pjoin(dpath, d)
            if isdir(arr_path):
                arr = list(xr.open_zarr(arr_path).values())[0]
                arr.data = darr.from_zarr(
                    os.path.join(arr_path, arr.name), inline_array=True
                )
                dslist.append(arr)
        if return_dict:
            ds = {d.name: d for d in dslist}
        else:
            ds = xr.merge(dslist, compat="no_conflicts")
    if (not return_dict) and post_process:
        ds = post_process(ds, dpath)
    return ds


def open_minian_mf(
    index_dims: List[str],
    result_format="xarray",
    pattern=r"minian$",
    sub_dirs: List[str] = [],
    exclude=True,
    **kwargs,
) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Open multiple minian datasets across multiple directories.

    This function recursively walks through directories under `dpath` and try to
    load minian datasets from all directories matching `pattern`. It will then
    combine them based on `index_dims` into either a `xr.Dataset` object or a
    `pd.DataFrame`. Optionally a subset of paths can be specified, so that they
    can either be excluded or white-listed. Additional keyword arguments will be
    passed directly to :func:`open_minian`.

    Parameters
    ----------
    dpath : str
        The root folder containing all datasets to be loaded.
    index_dims : List[str]
        List of dimensions that can be used to index and merge multiple
        datasets. All loaded datasets should have unique coordinates in the
        listed dimensions.
    result_format : str, optional
        If `"xarray"`, the result will be merged together recursively along each
        dimensions listed in `index_dims`. Users should make sure the
        coordinates are compatible and the merging will not cause generation of
        large NaN-padded results. If `"pandas"`, then a `pd.DataFrame` is
        returned, with columns corresponding to `index_dims` uniquely identify
        each dataset, and an additional column named "minian" of object dtype
        pointing to the loaded minian dataset objects. By default `"xarray"`.
    pattern : regexp, optional
        Pattern of minian dataset directory names. By default `r"minian$"`.
    sub_dirs : List[str], optional
        A list of sub-directories under `dpath`. Useful if only a subset of
        datasets under `dpath` should be recursively loaded. By default `[]`.
    exclude : bool, optional
        Whether to exclude directories listed under `sub_dirs`. If `True`, then
        any minian datasets under those specified in `sub_dirs` will be ignored.
        If `False`, then **only** the datasets under those specified in
        `sub_dirs` will be loaded (they still have to be under `dpath` though).
        by default `True`.

    Returns
    -------
    ds : Union[xr.Dataset, pd.DataFrame]
        The resulting combined datasets. If `result_format` is `"xarray"`, then
        a `xr.Dataset` will be returned, otherwise a `pd.DataFrame` will be
        returned.

    Raises
    ------
    NotImplementedError
        if `result_format` is not "xarray" or "pandas"
    """
    minian_dict = dict()
    for nextdir, dirlist, filelist in os.walk(dpath, topdown=False):
        nextdir = os.path.abspath(nextdir)
        cur_path = Path(nextdir)
        dir_tag = bool(
            (
                (any([Path(epath) in cur_path.parents for epath in sub_dirs]))
                or nextdir in sub_dirs
            )
        )
        if exclude == dir_tag:
            continue
        flist = list(filter(lambda f: re.search(pattern, f), filelist + dirlist))
        if flist:
            print("opening dataset under {}".format(nextdir))
            # if len(flist) > 1:
                # warnings.warn("multiple dataset found: {}".format(flist))
            fname = flist[-1]
            print("opening {}".format(fname))
            minian = open_minian(dpath=os.path.join(nextdir, fname), **kwargs)
            key = tuple([np.array_str(minian[d].values) for d in index_dims])
            minian_dict[key] = minian
            print(["{}: {}".format(d, v) for d, v in zip(index_dims, key)])

    if result_format == "xarray":
        return xrconcat_recursive(minian_dict, index_dims)
    elif result_format == "pandas":
        minian_df = pd.Series(minian_dict).rename("minian")
        minian_df.index.set_names(index_dims, inplace=True)
        return minian_df.to_frame()
    else:
        raise NotImplementedError("format {} not understood".format(result_format))


def save_minian(
    var: xr.DataArray,
    dpath: str,
    meta_dict: Optional[dict] = None,
    overwrite=False,
    chunks: Optional[dict] = None,
    compute=True,
    mem_limit="500MB",
) -> xr.DataArray:
    """
    Save a `xr.DataArray` with `zarr` storage backend following minian
    conventions.

    This function will store arbitrary `xr.DataArray` into `dpath` with `zarr`
    backend. A separate folder will be created under `dpath`, with folder name
    `var.name + ".zarr"`. Optionally metadata can be retrieved from directory
    hierarchy and added as coordinates of the `xr.DataArray`. In addition, an
    on-disk rechunking of the result can be performed using
    :func:`rechunker.rechunk` if `chunks` are given.

    Parameters
    ----------
    var : xr.DataArray
        The array to be saved.
    dpath : str
        The path to the minian dataset directory.
    meta_dict : dict, optional
        How metadata should be retrieved from directory hierarchy. The keys
        should be negative integers representing directory level relative to
        `dpath` (so `-1` means the immediate parent directory of `dpath`), and
        values should be the name of dimensions represented by the corresponding
        level of directory. The actual coordinate value of the dimensions will
        be the directory name of corresponding level. By default `None`.
    overwrite : bool, optional
        Whether to overwrite the result on disk. By default `False`.
    chunks : dict, optional
        A dictionary specifying the desired chunk size. The chunk size should be
        specified using :doc:`dask:array-chunks` convention, except the "auto"
        specifiication is not supported. The rechunking operation will be
        carried out with on-disk algorithms using :func:`rechunker.rechunk`. By
        default `None`.
    compute : bool, optional
        Whether to compute `var` and save it immediately. By default `True`.
    mem_limit : str, optional
        The memory limit for the on-disk rechunking algorithm, passed to
        :func:`rechunker.rechunk`. Only used if `chunks` is not `None`. By
        default `"500MB"`.

    Returns
    -------
    var : xr.DataArray
        The array representation of saving result. If `compute` is `True`, then
        the returned array will only contain delayed task of loading the on-disk
        `zarr` arrays. Otherwise all computation leading to the input `var` will
        be preserved in the result.

    Examples
    -------
    The following will save the variable `var` to directory
    `/spatial_memory/alpha/learning1/minian/important_array.zarr`, with the
    additional coordinates: `{"session": "learning1", "animal": "alpha",
    "experiment": "spatial_memory"}`.

    >>> save_minian(
    ...     var.rename("important_array"),
    ...     "/spatial_memory/alpha/learning1/minian",
    ...     {-1: "session", -2: "animal", -3: "experiment"},
    ... ) # doctest: +SKIP
    """
    dpath = os.path.normpath(dpath)
    Path(dpath).mkdir(parents=True, exist_ok=True)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.split(os.path.abspath(dpath))[0].split(os.sep)
        ds = ds.assign_coords(
            **dict([(dn, pathlist[di]) for dn, di in meta_dict.items()])
        )
    md = {True: "a", False: "w-"}[overwrite]
    fp = os.path.join(dpath, var.name + ".zarr")
    if overwrite:
        try:
            shutil.rmtree(fp)
        except FileNotFoundError:
            pass
    arr = ds.to_zarr(fp, compute=compute, mode=md)
    if (chunks is not None) and compute:
        chunks = {d: var.sizes[d] if v <= 0 else v for d, v in chunks.items()}
        dst_path = os.path.join(dpath, str(uuid4()))
        temp_path = os.path.join(dpath, str(uuid4()))
        with da.config.set(
            array_optimize=darr.optimization.optimize,
            delayed_optimize=default_delay_optimize,
        ):
            zstore = zr.open(fp)
            rechk = rechunker.rechunk(
                zstore[var.name], chunks, mem_limit, dst_path, temp_store=temp_path
            )
            rechk.execute()
        try:
            shutil.rmtree(temp_path)
        except FileNotFoundError:
            pass
        arr_path = os.path.join(fp, var.name)
        for f in os.listdir(arr_path):
            os.remove(os.path.join(arr_path, f))
        for f in os.listdir(dst_path):
            os.rename(os.path.join(dst_path, f), os.path.join(arr_path, f))
        os.rmdir(dst_path)
    if compute:
        arr = xr.open_zarr(fp)[var.name]
        arr.data = darr.from_zarr(os.path.join(fp, var.name), inline_array=True)
    return arr


def xrconcat_recursive(var: Union[dict, list], dims: List[str]) -> xr.Dataset:
    """
    Recursively concatenate `xr.DataArray` over multiple dimensions.

    Parameters
    ----------
    var : Union[dict, list]
        Either a `dict` or a `list` of `xr.DataArray` to be concatenated. If a
        `dict` then keys should be `tuple`, with length same as the length of
        `dims` and values corresponding to the coordinates that uniquely
        identify each `xr.DataArray`. If a `list` then each `xr.DataArray`
        should contain valid coordinates for each dimensions specified in
        `dims`.
    dims : List[str]
        Dimensions to be concatenated over.

    Returns
    -------
    ds : xr.Dataset
        The concatenated dataset.

    Raises
    ------
    NotImplementedError
        if input `var` is neither a `dict` nor a `list`
    """
    if len(dims) > 1:
        if type(var) is dict:
            var_dict = var
        elif type(var) is list:
            var_dict = {tuple([np.asscalar(v[d]) for d in dims]): v for v in var}
        else:
            raise NotImplementedError("type {} not supported".format(type(var)))
        try:
            var_dict = {k: v.to_dataset() for k, v in var_dict.items()}
        except AttributeError:
            pass
        data = np.empty(len(var_dict), dtype=object)
        for iv, ds in enumerate(var_dict.values()):
            data[iv] = ds
        index = pd.MultiIndex.from_tuples(list(var_dict.keys()), names=dims)
        var_ps = pd.Series(data=data, index=index)
        xr_ls = []
        for idx, v in var_ps.groupby(level=dims[0]):
            v.index = v.index.droplevel(dims[0])
            xarr = xrconcat_recursive(v.to_dict(), dims[1:])
            xr_ls.append(xarr)
        return xr.concat(xr_ls, dim=dims[0])
    else:
        if type(var) is dict:
            var = list(var.values())
        return xr.concat(var, dim=dims[0])


def update_meta(dpath, pattern=r"^minian\.nc$", meta_dict=None, backend="netcdf"):
    for dirpath, dirnames, fnames in os.walk(dpath):
        if backend == "netcdf":
            fnames = filter(lambda fn: re.search(pattern, fn), fnames)
        elif backend == "zarr":
            fnames = filter(lambda fn: re.search(pattern, fn), dirnames)
        else:
            raise NotImplementedError("backend {} not supported".format(backend))
        for fname in fnames:
            f_path = os.path.join(dirpath, fname)
            pathlist = os.path.normpath(dirpath).split(os.sep)
            new_ds = xr.Dataset()
            old_ds = open_minian(f_path, f_path, backend)
            new_ds.attrs = deepcopy(old_ds.attrs)
            old_ds.close()
            new_ds = new_ds.assign_coords(
                **dict(
                    [(cdname, pathlist[cdval]) for cdname, cdval in meta_dict.items()]
                )
            )
            if backend == "netcdf":
                new_ds.to_netcdf(f_path, mode="a")
            elif backend == "zarr":
                new_ds.to_zarr(f_path, mode="w")
            print("updated: {}".format(f_path))


def get_chk(arr: xr.DataArray) -> dict:
    """
    Get chunks of a `xr.DataArray`.

    Parameters
    ----------
    arr : xr.DataArray
        The input `xr.DataArray`

    Returns
    -------
    chk : dict
        Dictionary mapping dimension names to chunks.
    """
    return {d: c for d, c in zip(arr.dims, arr.chunks)}


def rechunk_like(x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
    """
    Rechunk the input `x` such that its chunks are compatible with `y`.

    Parameters
    ----------
    x : xr.DataArray
        The array to be rechunked.
    y : xr.DataArray
        The array where chunk information are extracted.

    Returns
    -------
    x_chk : xr.DataArray
        The rechunked `x`.
    """
    try:
        dst_chk = get_chk(y)
        comm_dim = set(x.dims).intersection(set(dst_chk.keys()))
        dst_chk = {d: max(dst_chk[d]) for d in comm_dim}
        return x.chunk(dst_chk)
    except TypeError:
        return x.compute()


def get_optimal_chk(
    arr: xr.DataArray,
    dim_grp=[("frame",), ("height", "width")],
    csize=256,
    dtype: Optional[type] = None,
) -> dict:
    """
    Compute the optimal chunk size across all dimensions of the input array.

    This function use `dask` autochunking mechanism to determine the optimal
    chunk size of an array. The difference between this and directly using
    "auto" as chunksize is that it understands which dimensions are usually
    chunked together with the help of `dim_grp`. It also support computing
    chunks for custom `dtype` and explicit requirement of chunk size.

    Parameters
    ----------
    arr : xr.DataArray
        The input array to estimate for chunk size.
    dim_grp : list, optional
        List of tuples specifying which dimensions are usually chunked together
        during computation. For each tuple in the list, it is assumed that only
        dimensions in the tuple will be chunked while all other dimensions in
        the input `arr` will not be chunked. Each dimensions in the input `arr`
        should appear once and only once across the list. By default
        `[("frame",), ("height", "width")]`.
    csize : int, optional
        The desired space each chunk should occupy, specified in MB. By default
        `256`.
    dtype : type, optional
        The datatype of `arr` during actual computation in case that will be
        different from the current `arr.dtype`. By default `None`.

    Returns
    -------
    chk : dict
        Dictionary mapping dimension names to chunk sizes.
    """
    if dtype is not None:
        arr = arr.astype(dtype)
    dims = arr.dims
    if not dim_grp:
        dim_grp = [(d,) for d in dims]
    chk_compute = dict()
    for dg in dim_grp:
        d_rest = set(dims) - set(dg)
        dg_dict = {d: "auto" for d in dg}
        dr_dict = {d: -1 for d in d_rest}
        dg_dict.update(dr_dict)
        with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
            arr_chk = arr.chunk(dg_dict)
        chk = get_chunksize(arr_chk)
        chk_compute.update({d: chk[d] for d in dg})
    with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
        arr_chk = arr.chunk({d: "auto" for d in dims})
    chk_store_da = get_chunksize(arr_chk)
    chk_store = dict()
    for d in dims:
        ncomp = int(arr.sizes[d] / chk_compute[d])
        sz = np.array(factors(ncomp)) * chk_compute[d]
        chk_store[d] = sz[np.argmin(np.abs(sz - chk_store_da[d]))]
    return chk_compute, chk_store_da


def get_chunksize(arr: xr.DataArray) -> dict:
    """
    Get chunk size of a `xr.DataArray`.

    Parameters
    ----------
    arr : xr.DataArray
        The input `xr.DataArray`.

    Returns
    -------
    chk : dict
        Dictionary mapping dimension names to chunk sizes.
    """
    dims = arr.dims
    sz = arr.data.chunksize
    return {d: s for d, s in zip(dims, sz)}


def factors(x: int) -> List[int]:
    """
    Compute all factors of an interger.

    Parameters
    ----------
    x : int
        Input

    Returns
    -------
    factors : List[int]
        List of factors of `x`.
    """
    return [i for i in range(1, x + 1) if x % i == 0]

def get_noise_fft(
    varr: xr.DataArray, noise_range=(0.25, 0.5), noise_method="logmexp"
) -> xr.DataArray:
    """
    Estimates noise along the "frame" dimension aggregating power spectral
    density within `noise_range`.

    This function compute a Fast Fourier transform (FFT) along the "frame"
    dimension in a vectorized fashion, and estimate noise by aggregating its
    power spectral density (PSD). Note that `noise_range` is specified relative
    to the sampling frequency, so 0.5 represents the Nyquist frequency. Three
    `noise_method` are availabe for aggregating the psd: "mean" and "median"
    will use the mean and median across all frequencies as the estimation of
    noise. "logmexp" takes the mean of the logarithmic psd, then transform it
    back with an exponential function.

    Parameters
    ----------
    varr : xr.DataArray
        Input data, should have a "frame" dimension.
    noise_range : tuple, optional
        Range of noise frequency to be aggregated as a fraction of sampling
        frequency. By default `(0.25, 0.5)`.
    noise_method : str, optional
        Method of aggreagtion for noise. Should be one of `"mean"` `"median"`
        `"logmexp"` or `"sum"`. By default `"logmexp"`.

    Returns
    -------
    sn : xr.DataArray
        Spectral density of the noise. Same shape as `varr` with the "frame"
        dimension removed.
    """
    try:
        clt = get_client()
        threads = min(clt.nthreads().values())
    except ValueError:
        threads = 1
    sn = xr.apply_ufunc(
        noise_fft,
        varr,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=True,
        kwargs=dict(
            noise_range=noise_range, noise_method=noise_method, threads=threads
        ),
        output_dtypes=[np.float32],
    )
    return sn


def noise_fft(
    px: np.ndarray, noise_range=(0.25, 0.5), noise_method="logmexp", threads=1
) -> float:
    """
    Estimates noise of the input by aggregating power spectral density within
    `noise_range`.

    The PSD is estimated using FFT.

    Parameters
    ----------
    px : np.ndarray
        Input data.
    noise_range : tuple, optional
        Range of noise frequency to be aggregated as a fraction of sampling
        frequency. By default `(0.25, 0.5)`.
    noise_method : str, optional
        Method of aggreagtion for noise. Should be one of `"mean"` `"median"`
        `"logmexp"` or `"sum"`. By default "logmexp".
    threads : int, optional
        Number of threads to use for pyfftw. By default `1`.

    Returns
    -------
    noise : float
        The estimated noise level of input.

    See Also
    -------
    get_noise_fft
    """
    _T = len(px)
    nr = np.around(np.array(noise_range) * _T).astype(int)
    px = 1 / _T * np.abs(numpy_fft.rfft(px, threads=threads)[nr[0] : nr[1]]) ** 2
    if noise_method == "mean":
        return np.sqrt(px.mean())
    elif noise_method == "median":
        return np.sqrt(px.median())
    elif noise_method == "logmexp":
        eps = np.finfo(px.dtype).eps
        return np.sqrt(np.exp(np.log(px + eps).mean()))
    elif noise_method == "sum":
        return np.sqrt(px.sum())


def get_noise_welch(
    varr: xr.DataArray, noise_range=(0.25, 0.5), noise_method="logmexp"
) -> xr.DataArray:
    """
    Estimates noise along the "frame" dimension aggregating power spectral
    density within `noise_range`.

    The PSD is estimated using welch method as an alternative to FFT. The welch
    method assumes the noise in the signal to be a stochastic process and
    attenuates noise by windowing the original signal into segments and
    averaging over them.

    Parameters
    ----------
    varr : xr.DataArray
        Input data. Should have a "frame" dimension.
    noise_range : tuple, optional
        Range of noise frequency to be aggregated as a fraction of sampling
        frequency. By default `(0.25, 0.5)`.
    noise_method : str, optional
        Method of aggreagtion for noise. Should be one of `"mean"` `"median"`
        `"logmexp"` or `"sum"`. By default `"logmexp"`.

    Returns
    -------
    sn : xr.DataArray
        Spectral density of the noise. Same shape as `varr` with the "frame"
        dimension removed.

    See Also
    -------
    get_noise_fft : For more details on the parameters.
    """
    sn = xr.apply_ufunc(
        noise_welch,
        varr.chunk(dict(frame=-1)),
        input_core_dims=[["frame"]],
        dask="parallelized",
        vectorize=True,
        kwargs=dict(noise_range=noise_range, noise_method=noise_method),
        output_dtypes=[varr.dtype],
    )
    return sn


def noise_welch(
    y: np.ndarray, noise_range=(0.25, 0.5), noise_method="logmexp"
) -> float:
    """
    Estimates noise of the input by aggregating power spectral density within
    `noise_range`.

    The PSD is estimated using welch method.

    Parameters
    ----------
    px : np.ndarray
        Input data.
    noise_range : tuple, optional
        Range of noise frequency to be aggregated as a fraction of sampling
        frequency. By default `(0.25, 0.5)`.
    noise_method : str, optional
        Method of aggreagtion for noise. Should be one of `"mean"` `"median"`
        `"logmexp"` or `"sum"`. By default `"logmexp"`.
    threads : int, optional
        Number of threads to use for pyfftw. By default `1`.

    Returns
    -------
    noise : float
        The estimated noise level of input.

    See Also
    -------
    get_noise_welch
    """
    ff, Pxx = welch(y)
    mask0, mask1 = ff > noise_range[0], ff < noise_range[1]
    mask = np.logical_and(mask0, mask1)
    Pxx_ind = Pxx[mask]
    sn = {
        "mean": lambda x: np.sqrt(np.mean(x / 2)),
        "median": lambda x: np.sqrt(np.median(x / 2)),
        "logmexp": lambda x: np.sqrt(np.exp(np.mean(np.log(x / 2)))),
    }[noise_method](Pxx_ind)
    return sn


def update_spatial(
    Y: xr.DataArray,
    A: xr.DataArray,
    C: xr.DataArray,
    sn: xr.DataArray,
    b: xr.DataArray = None,
    f: xr.DataArray = None,
    dl_wnd=5,
    sparse_penal=0.5,
    update_background=False,
    normalize=True,
    size_thres=(9, None),
    in_memory=False,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Update spatial components given the input data and temporal dynamic for each
    cell.

    This function carries out spatial update of the CNMF algorithm. The update
    is done in parallel and independently for each pixel. To save computation
    time, we compute a subsetting matrix `sub` by dilating the initial
    spatial foorprint of each cell. The window size of the dilation is
    controled by `dl_wnd`. Then for each pixel, only cells that have a non-zero
    value in `sub` at the current pixel will be considered for update.
    Optionally, the spatial footprint of the background can be updated in the
    same fashion based on the temporal dynamic of the background. After the
    update, the spatial footprint of each cell can be optionally noramlized to
    unit sum, so that difference in fluorescent intensity will not be reflected
    in spatial footprint. A `size_thres` can be passed in to filter out cells
    whose size (number of non-zero values in spatial footprint) is outside the
    specified range. Finally, the temporal dynamic of cells `C` can either be
    load in memory before the update or lazy-loaded during the update. Note that
    if `in_memory` is `False`, then `C` must be stored under the intermediate
    folder specified as environment variable `MINIAN_INTERMEDIATE`.

    Parameters
    ----------
    Y : xr.DataArray
        Input movie data. Should have dimensions "height", "width" and "frame".
    A : xr.DataArray
        Previous estimation of spatial footprints. Should have dimension
        "height", "width" and "unit_id".
    C : xr.DataArray
        Estimation of temporal component for each cell. Should have dimension
        "frame" and "unit_id".
    sn : xr.DataArray
        Estimation of noise level for each pixel. Should have dimension "height"
        and "width".
    b : xr.DataArray, optional
        Previous estimation of spatial footprint of background. Fhould have
        dimension "height" and "width".
    f : xr.DataArray, optional
        Estimation of temporal dynamic of background. Should have dimension
        "frame".
    dl_wnd : int, optional
        Window of morphological dilation in pixel when computing the subsetting
        matrix. By default `5`.
    sparse_penal : float, optional
        Global scalar controlling sparsity of the result. The higher the value,
        the sparser the spatial footprints. By default `0.5`.
    update_background : bool, optional
        Whether to update the spatial footprint of background. If `True`, then
        both `b` and `f` need to be provided. By default `False`.
    normalize : bool, optional
        Whether to normalize resulting spatial footprints of each cell to unit
        sum. By default `True`
    size_thres : tuple, optional
        The range of size in pixel allowed for the resulting spatial footprints.
        If `None`, then no filtering will be done. By default `(9, None)`.
    in_memory : bool, optional
        Whether to load `C` into memory before spatial update. By default
        `False`.

    Returns
    -------
    A_new : xr.DataArray
        New estimation of spatial footprints. Same shape as `A` except the
        "unit_id" dimension might be smaller due to filtering.
    mask : xr.DataArray
        Boolean mask of whether a cell passed size filtering. Has dimension
        "unit_id" that is same as input `A`. Useful for subsetting other
        variables based on the result of spatial update.
    b_new : xr.DataArray
        New estimation of spatial footprint of background. Only returned if
        `update_background` is `True`. Same shape as `b`.
    norm_fac : xr.DataArray
        Normalizing factor. Userful to scale temporal activity of cells. Only
        returned if `normalize` is `True`.
    Notes
    -------
    During spatial update, the algorithm solve the following optimization
    problem for each pixel:

    .. math::
        \\begin{aligned}
        & \\underset{\mathbf{a}}{\\text{minimize}}
        & & \\left \\lVert \mathbf{y} - \mathbf{a}^T \mathbf{C} \\right \\rVert
        ^2 + \\alpha \\left \\lvert \mathbf{a} \\right \\rvert \\\\
        & \\text{subject to} & & \mathbf{a} \geq 0
        \\end{aligned}

    Where :math:`\mathbf{y}` is the fluorescent dynamic of the pixel,
    :math:`\mathbf{a}` is spatial footprint values across all cells on that
    pixel, :math:`\mathbf{C}` is temporal component matrix across all cells. The
    parameter :math:`\\alpha` is the product of the noise level on each pixel
    `sn` and the global scalar `sparse_penal`. Higher value of :math:`\\alpha`
    will result in more sparse estimation of spatial footprints.
    """
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    if in_memory:
        C_store = C.compute().values
    else:
        C_path = os.path.join(intpath, C.name + ".zarr", C.name)
        C_store = zarr.open_array(C_path)
    print("estimating penalty parameter")
    alpha = sparse_penal * sn
    alpha = rechunk_like(alpha.compute(), sn)
    print("computing subsetting matrix")
    selem = moph.disk(dl_wnd)
    sub = xr.apply_ufunc(
        cv2.dilate,
        A,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        kwargs=dict(kernel=selem),
        dask="parallelized",
        output_dtypes=[A.dtype],
    )
    sub = sub > 0
    sub.data = sub.data.map_blocks(sparse.COO)
    if update_background:
        assert b is not None, "`b` must be provided when updating background"
        assert f is not None, "`f` must be provided when updating background"
        b_in = rechunk_like(b > 0, Y).assign_coords(unit_id=-1).expand_dims("unit_id")
        b_in.data = b_in.data.map_blocks(sparse.COO)
        b_in = b_in.compute()
        sub = xr.concat([sub, b_in], "unit_id")
        f_in = f.compute().data
    else:
        f_in = None
    sub = rechunk_like(sub.transpose("height", "width", "unit_id").compute(), Y)
    print("fitting spatial matrix")
    ssub = darr.map_blocks(
        sps_any,
        sub.data,
        drop_axis=2,
        chunks=((1, 1)),
        meta=sparse.ones(1).astype(bool),
    ).compute()
    Y_trans = Y.transpose("height", "width", "frame")
    # take fast route if a lot of chunks are empty
    if ssub.sum() < 500:
        A_new = np.empty(sub.data.numblocks, dtype=object)
        for (hblk, wblk), has_unit in np.ndenumerate(ssub):
            cur_sub = sub.data.blocks[hblk, wblk, :]
            if has_unit:
                cur_blk = update_spatial_block(
                    Y_trans.data.blocks[hblk, wblk, :],
                    alpha.data.blocks[hblk, wblk],
                    cur_sub,
                    C_store=C_store,
                    f=f_in,
                )
            else:
                cur_blk = darr.array(sparse.zeros((cur_sub.shape)))
            A_new[hblk, wblk, 0] = cur_blk
        A_new = darr.block(A_new.tolist())
    else:
        A_new = update_spatial_block(
            Y_trans.data,
            alpha.data,
            sub.data,
            C_store=C_store,
            f=f_in,
        )
    with da.config.set(**{"optimization.fuse.ave-width": 6}):
        A_new = da.optimize(A_new)[0]
    A_new = xr.DataArray(
        darr.moveaxis(A_new, -1, 0).map_blocks(lambda a: a.todense(), dtype=A.dtype),
        dims=["unit_id", "height", "width"],
        coords={
            "unit_id": sub.coords["unit_id"],
            "height": A.coords["height"],
            "width": A.coords["width"],
        },
    )
    A_new = save_minian(
        A_new.rename("A_new"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    add_rets = []
    if update_background:
        b_new = A_new.sel(unit_id=-1).compute()
        A_new = A_new[:-1, :, :]
        add_rets.append(b_new)
    if size_thres:
        low, high = size_thres
        A_bin = A_new > 0
        mask = np.ones(A_new.sizes["unit_id"], dtype=bool)
        if low:
            mask = np.logical_and(
                (A_bin.sum(["height", "width"]) > low).compute(), mask
            )
        if high:
            mask = np.logical_and(
                (A_bin.sum(["height", "width"]) < high).compute(), mask
            )
        mask = xr.DataArray(
            mask, dims=["unit_id"], coords={"unit_id": A_new.coords["unit_id"].values}
        )
    else:
        mask = (A_new.sum(["height", "width"]) > 0).compute()
    print("{} out of {} units dropped".format(len(mask) - mask.sum().values, len(mask)))
    A_new = A_new.sel(unit_id=mask)
    if normalize:
        norm_fac = A_new.max(["height", "width"]).compute()
        A_new = A_new / norm_fac
        add_rets.append(norm_fac)
    return (A_new, mask, *add_rets)


def sps_any(x: sparse.COO) -> np.ndarray:
    """
    Compute `any` on a sparse array.

    Parameters
    ----------
    x : sparse.COO
        Input sparse array.

    Returns
    -------
    x_any : np.ndarray
        2d boolean numpy array.
    """
    return np.atleast_2d(x.nnz > 0)


def update_spatial_perpx(
    y: np.ndarray,
    alpha: float,
    sub: sparse.COO,
    C_store: Union[np.ndarray, zarr.core.Array],
    f: Optional[np.ndarray],
) -> sparse.COO:
    """
    Update spatial footprints across all the cells for a single pixel.

    This function use :class:`sklearn.linear_model.LassoLars` to solve the
    optimization problem. `C_store` can either be a in-memory numpy array, or a
    zarr array, in which case it will be lazy-loaded. If `f` is not `None`, then
    `sub[-1]` is expected to be the subsetting mask for background, and the last
    element of the return value will be the spatial footprint of background.

    Parameters
    ----------
    y : np.ndarray
        Input fluorescent trace for the given pixel.
    alpha : float
        Parameter of the optimization problem controlling sparsity.
    sub : sparse.COO
        Subsetting matrix.
    C_store : Union[np.ndarray, zarr.core.Array]
        Estimation of temporal dynamics of cells.
    f : np.ndarray, optional
        Temporal dynamic of background.

    Returns
    -------
    A_px : sparse.COO
        Spatial footprint values across all cells for the given pixel.

    See Also
    -------
    update_spatial : for more explanation of parameters
    """
    if f is not None:
        idx = sub[:-1].nonzero()[0]
    else:
        idx = sub.nonzero()[0]
    try:
        C = C_store.get_orthogonal_selection((idx, slice(None))).T
    except AttributeError:
        C = C_store[idx, :].T
    if (f is not None) and sub[-1]:
        C = np.concatenate([C, f.reshape((-1, 1))], axis=1)
        idx = np.concatenate([idx, np.array(len(sub) - 1).reshape(-1)])
    clf = LassoLars(alpha=alpha, positive=True)
    coef = clf.fit(C, y).coef_
    mask = coef > 0
    coef = coef[mask]
    idx = idx[mask]
    return sparse.COO(coords=idx, data=coef, shape=sub.shape)


@darr.as_gufunc(signature="(f),(),(u)->(u)", output_dtypes=float)
def update_spatial_block(
    y: np.ndarray, alpha: np.ndarray, sub: sparse.COO, **kwargs
) -> sparse.COO:
    """
    Carry out spatial update for each 3d block of data.

    This function wraps around :func:`update_spatial_perpx` so that it can be
    applied to 3d blocks of data. Keyword arguments are passed to
    :func:`update_spatial_perpx`.

    Parameters
    ----------
    y : np.ndarray
        Input data, should have dimension (height, width, frame).
    alpha : np.ndarray
        Alpha parameter for the optimization problem. Should have dimension
        (height, width).
    sub : sparse.COO
        Subsetting matrix. Should have dimension (height, width, unit_id).

    Returns
    -------
    A_blk : sparse.COO
        Resulting spatial footprints. Should have dimension (height, width,
        unit_id).

    See Also
    -------
    update_spatial_perpx
    update_spatial
    """
    C_store = kwargs.get("C_store")
    f = kwargs.get("f")
    crd_ls = []
    data_ls = []
    for h, w in zip(*sub.any(axis=-1).nonzero()):
        res = update_spatial_perpx(y[h, w, :], alpha[h, w], sub[h, w, :], C_store, f)
        crd = res.coords
        crd = np.concatenate([np.full_like(crd, h), np.full_like(crd, w), crd], axis=0)
        crd_ls.append(crd)
        data_ls.append(res.data)
    if data_ls:
        return sparse.COO(
            coords=np.concatenate(crd_ls, axis=1),
            data=np.concatenate(data_ls),
            shape=sub.shape,
        )
    else:
        return sparse.zeros(sub.shape)


def compute_trace(
    Y: xr.DataArray, A: xr.DataArray, b: xr.DataArray, C: xr.DataArray, f: xr.DataArray
) -> xr.DataArray:
    """
    Compute the residule traces `YrA` for each cell.

    `YrA` is computed as `C + A_norm(YtA - CtA)`, where `YtA` is `(Y -
    b.dot(f)).tensordot(A, ["height", "width"])`, representing the projection of
    background-subtracted movie onto the spatial footprints, and `CtA` is
    `C.dot(AtA, ["unit_id"])` with `AtA = A.tensordot(A, ["height", "width"])`,
    hence `CtA` represent for each cell the sum of temporal activities that's
    shared with any other cells, then finally `A_norm` is a "unit_id"x"unit_id"
    diagonal matrix that normalize the result with sum of squares of spatial
    footprints for each cell. Together, the `YrA` trace is a "unit_id"x"frame"
    matrix, representing the sum of previous temporal components and the
    residule temporal fluctuations as estimated by projecting the data onto the
    spatial footprints and subtracting the cross-talk fluctuations.

    Parameters
    ----------
    Y : xr.DataArray
        Input movie data. Should have dimensions ("frame", "height", "width").
    A : xr.DataArray
        Spatial footprints of cells. Should have dimensions ("unit_id", "height",
        "width").
    b : xr.DataArray
        Spatial footprint of background. Should have dimensions ("height", "width").
    C : xr.DataArray
        Temporal components of cells. Should have dimensions ("frame", "unit_id").
    f : xr.DataArray
        Temporal dynamic of background. Should have dimension "frame".

    Returns
    -------
    YrA : xr.DataArray
        Residule traces for each cell. Should have dimensions("frame", "unit_id").
    """
    fms = Y.coords["frame"]
    uid = A.coords["unit_id"]
    Y = Y.data
    A = darr.from_array(A.data.map_blocks(sparse.COO).compute(), chunks=-1)
    C = C.data.map_blocks(sparse.COO).T
    b = (
        b.fillna(0)
        .data.map_blocks(sparse.COO)
        .reshape((1, Y.shape[1], Y.shape[2]))
        .compute()
    )
    f = f.fillna(0).data.reshape((-1, 1))
    AtA = darr.tensordot(A, A, axes=[(1, 2), (1, 2)]).compute()
    A_norm = (
        (1 / (A ** 2).sum(axis=(1, 2)))
        .map_blocks(
            lambda a: sparse.diagonalize(sparse.COO(a)), chunks=(A.shape[0], A.shape[0])
        )
        .compute()
    )
    B = darr.tensordot(f, b, axes=[(1), (0)])
    Y = Y - B
    YtA = darr.tensordot(Y, A, axes=[(1, 2), (1, 2)])
    YtA = darr.dot(YtA, A_norm)
    CtA = darr.dot(C, AtA)
    CtA = darr.dot(CtA, A_norm)
    YrA = (YtA - CtA + C).clip(0)
    arr_opt = fct.partial(
        custom_arr_optimize,
        inline_patterns=["from-getitem-transpose"],
        rename_dict={"tensordot": "tensordot_restricted"},
    )
    with da.config.set(array_optimize=arr_opt):
        YrA = da.optimize(YrA)[0]
    YrA = xr.DataArray(
        YrA,
        dims=["frame", "unit_id"],
        coords={"frame": fms, "unit_id": uid},
    )
    return YrA.transpose("unit_id", "frame")


def update_temporal(
    A: xr.DataArray,
    C: xr.DataArray,
    b: Optional[xr.DataArray] = None,
    f: Optional[xr.DataArray] = None,
    Y: Optional[xr.DataArray] = None,
    YrA: Optional[xr.DataArray] = None,
    noise_freq=0.25,
    p=2,
    add_lag="p",
    jac_thres=0.1,
    sparse_penal=1,
    bseg: Optional[np.ndarray] = None,
    med_wd: Optional[int] = None,
    zero_thres=1e-8,
    max_iters=200,
    use_smooth=True,
    normalize=True,
    warm_start=False,
    post_scal=False,
    scs_fallback=False,
    concurrent_update=False,
) -> Tuple[
    xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray
]:
    """
    Update temporal components and deconvolve calcium traces for each cell given
    spatial footprints.

    This function carries out temporal update of the CNMF algorithm. The update
    is done in parallel and independently for each group of cells. The grouping
    of cells is controlled by `jac_thres`. The relationship between calcium and
    deconvolved spikes is modeled as an Autoregressive process (AR) of order
    `p`. The AR coefficients are estimated from autocovariances of `YrA` traces
    for each cell, with `add_lag` controls how many timesteps of autocovariances
    are used. Optionally, the `YrA` traces can be smoothed for the estimation of
    AR coefficients only. The noise level for each cell is estimated using FFT
    with `noise_freq` as cut-off, and controls the sparsity of the result
    together with the global `sparse_penal` parameter. `YrA` traces for each
    cells can be optionally normalized to unit sum to make `sparse_penal` to
    have comparable effects across cells. If abrupt change of baseline
    fluorescence is expected, a `bseg` vector can be passed to enable estimation
    of independent baseline for different segments of time. The temporal update
    itself is performed by solving an optimization problem using `cvxpy`, with
    `concurrent_update`, `warm_start`, `max_iters`, `scs_fallback` controlling
    different aspects of the optimization. Finally, the results can be filtered
    with `zero_thres` to suppress small values caused by numerical errors, and a
    post-hoc scaling process can be optionally used to scale the result based on
    `YrA` to get around unwanted effects from sparse penalty or normalization.

    Parameters
    ----------
    A : xr.DataArray
        Estimation of spatial footprints for each cell. Should have dimensions
        ("unit_id", "height", "width").
    C : xr.DataArray
        Previous estimation of calcium dynamic of cells. Should have dimensions
        ("frame", "unit_id"). Only used if `warm_start = True` or if `YrA is
        None`.
    b : xr.DataArray, optional
        Estimation of spatial footprint of background. Should have dimensions
        ("height", "width"). Only used if `YrA is None`. By default `None`.
    f : xr.DataArray, optional
        Estimation of temporal dynamic of background. Should have dimension
        "frame". Only used if `YrA is None`. By default `None`.
    Y : xr.DataArray, optional
        Input movie data. Should have dimensions ("frame", "height", "width").
        Only used if `YrA is None`. By default `None`.
    YrA : xr.DataArray, optional
        Estimation of residule traces for each cell. Should have dimensions
        ("frame", "unit_id"). If `None` then one will be computed using
        `computea_trace` with relevant inputs. By default `None`.
    noise_freq : float, optional
        Frequency cut-off for both the estimation of noise level and the
        optional smoothing, specified as a fraction of sampling frequency. By
        default `0.25`.
    p : int, optional
        Order of the AR process. By default `2`.
    add_lag : str, optional
        Additional number of timesteps in covariance to use for the estimation
        of AR coefficients. If `0`, then only the first `p` number of timesteps
        will be used to estimate the `p` number of AR coefficients. If greater
        than `0`, then the system is over-determined and least square will be
        used to estimate AR coefficients. If `"p"`, then `p` number of
        additional timesteps will be used. By default `"p"`.
    jac_thres : float, optional
        Threshold for Jaccard Index. Cells whose overlap in spatial footprints
        (number of common pixels divided by number of total pixels) exceeding
        this threshold will be grouped together transitively for temporal
        update. By default `0.1`.
    sparse_penal : int, optional
        Global scalar controlling sparsity of the result. The higher the value,
        the sparser the deconvolved spikes. By default `1`.
    bseg : np.ndarray, optional
        1d vector with length "frame" representing segments for which baseline
        should be estimated independently. An independent baseline will be
        estimated for frames corresponding to each unique label in this vector.
        If `None` then a single scalar baseline will be estimated for each cell.
        By default `None`.
    med_wd : int, optional
        Window size for the median filter used for baseline correction. For each
        cell, the baseline flurescence is estimated by median-filtering the
        temporal activity. Then the baseline is subtracted from the temporal
        activity right before the optimization step. If `None` then no baseline
        correction will be performed. By default `None`.
    zero_thres : float, optional
        Threshold to filter out small values in the result. Any values smaller
        than this threshold will be set to zero. By default `1e-8`.
    max_iters : int, optional
        Maximum number of iterations for optimization. Can be increased to get
        around sub-optimal results. By default `200`.
    use_smooth : bool, optional
        Whether to smooth the `YrA` for the estimation of AR coefficients. If
        `True`, then a smoothed version of `YrA` will be computed by low-pass
        filter with `noise_freq` and used for the estimation of AR coefficients
        only. By default `True`.
    normalize : bool, optional
        Whether to normalize `YrA` for each cell to unit sum such that sparse
        penalty has simlar effect for all the cells. Each group of cell will be
        normalized together (with mean of the sum for each cell) to preserve
        relative amplitude of fluorescence between overlapping cells. By default
        `True`.
    warm_start : bool, optional
        Whether to use previous estimation of `C` to warm start the
        optimization. Can lead to faster convergence in theory. Experimental. By
        default `False`.
    post_scal : bool, optional
        Whether to apply the post-hoc scaling process, where a scalar will be
        estimated with least square for each cell to scale the amplitude of
        temporal component to `YrA`. Useful to get around unwanted dampening of
        result values caused by high `sparse_penal` or to revert the per-cell
        normalization. By default `False`.
    scs_fallback : bool, optional
        Whether to fall back to `scs` solver if the default `ecos` solver fails.
        By default `False`.
    concurrent_update : bool, optional
        Whether to update a group of cells as a single optimization problem.
        Yields slightly more accurate estimation when cross-talk between cells
        are severe, but significantly increase convergence time and memory
        demand. By default `False`.

    Returns
    -------
    C_new : xr.DataArray
        New estimation of the calcium dynamic for each cell. Should have same
        shape as `C` except the "unit_id" dimension might be smaller due to
        dropping of cells and filtering.
    S_new : xr.DataArray
        New estimation of the deconvolved spikes for each cell. Should have
        dimensions ("frame", "unit_id") and same shape as `C_new`.
    b0_new : xr.DataArray
        New estimation of baseline fluorescence for each cell. Should have
        dimensions ("frame", "unit_id") and same shape as `C_new`. Each cell
        should only have one unique value if `bseg is None`.
    c0_new : xr.DataArray
        New estimation of a initial calcium decay, in theory triggered by
        calcium events happened before the recording starts. Should have
        dimensions ("frame", "unit_id") and same shape as `C_new`.
    g : xr.DataArray
        Estimation of AR coefficient for each cell. Useful for visualizing
        modeled calcium dynamic. Should have dimensions ("lag", "unit_id") with
        "lag" having length `p`.
    mask : xr.DataArray
        Boolean mask of whether a cell has any temporal dynamic after the update
        and optional filtering. Has dimension "unit_id" that is same as input
        `C`. Useful for subsetting other variables based on the result of
        temporal update.


    Notes
    -------
    During temporal update, the algorithm solve the following optimization
    problem for each cell:

    .. math::
        \\begin{aligned}
        & \\underset{\mathbf{c} \, \mathbf{b_0} \,
        \mathbf{c_0}}{\\text{minimize}}
        & & \\left \\lVert \mathbf{y} - \mathbf{c} - \mathbf{c_0} -
        \mathbf{b_0} \\right \\rVert ^2 + \\alpha \\left \\lvert \mathbf{G}
        \mathbf{c} \\right \\rvert \\\\
        & \\text{subject to}
        & & \mathbf{c} \geq 0, \; \mathbf{G} \mathbf{c} \geq 0 
        \\end{aligned}

    Where :math:`\mathbf{y}` is the estimated residule trace (`YrA`) for the
    cell, :math:`\mathbf{c}` is the calcium dynamic of the cell,
    :math:`\mathbf{G}` is a "frame"x"frame" matrix constructed from the
    estimated AR coefficients of cell, such that the deconvolved spikes of the
    cell is given by :math:`\mathbf{G}\mathbf{c}`. If `bseg is None`, then
    :math:`\mathbf{b_0}` is a single scalar, otherwise it is a 1d vector with
    dimension "frame" constrained to have multiple independent values, each
    corresponding to a segment of time specified in `bseg`. :math:`\mathbf{c_0}`
    is a 1d vector with dimension "frame" constrained to be the product of a
    scalar (representing initial calcium concentration) and the decay dynamic
    given by the estimated AR coefficients. The parameter :math:`\\alpha` is the
    product of estimated noise level of the cell and the global scalar
    `sparse_penal`. Higher value of :math:`\\alpha` will result in more sparse
    estimation of deconvolved spikes.
    """
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    if YrA is None:
        YrA = compute_trace(Y, A, b, C, f).persist()
    Ymask = (YrA > 0).any("frame").compute()
    A, C, YrA = A.sel(unit_id=Ymask), C.sel(unit_id=Ymask), YrA.sel(unit_id=Ymask)
    print("grouping overlaping units")
    A_sps = (A.data.map_blocks(sparse.COO) > 0).compute().astype(np.float32)
    A_inter = sparse.tensordot(A_sps, A_sps, axes=[(1, 2), (1, 2)])
    A_usum = np.tile(A_sps.sum(axis=(1, 2)).todense(), (A_sps.shape[0], 1))
    A_usum = A_usum + A_usum.T
    jac = scipy.sparse.csc_matrix(A_inter / (A_usum - A_inter) > jac_thres)
    unit_labels = label_connected(jac)
    YrA = YrA.assign_coords(unit_labels=("unit_id", unit_labels))
    print("updating temporal components")
    c_ls = []
    s_ls = []
    b_ls = []
    c0_ls = []
    g_ls = []
    uid_ls = []
    grp_dim = "unit_labels"
    C = C.assign_coords(unit_labels=("unit_id", unit_labels))
    if warm_start:
        C.data = C.data.map_blocks(scipy.sparse.csr_matrix)
    inline_opt = fct.partial(
        custom_delay_optimize,
        inline_patterns=["getitem", "rechunk-merge"],
    )
    for cur_YrA, cur_C in zip(YrA.groupby(grp_dim), C.groupby(grp_dim)):
        uid_ls.append(cur_YrA[1].coords["unit_id"].values.reshape(-1))
        cur_YrA, cur_C = cur_YrA[1].data.rechunk(-1), cur_C[1].data.rechunk(-1)
        # peak memory demand for cvxpy is roughly 500 times input
        mem_cvx = cur_YrA.nbytes if concurrent_update else cur_YrA[0].nbytes
        mem_cvx = mem_cvx * 500
        mem_demand = max(mem_cvx, cur_YrA.nbytes * 5) / 1e6
        # issue a warning if expected memory demand is larger than 1G
        if mem_demand > 1e3:
            warnings.warn(
                "{} cells will be updated togeter, "
                "which takes roughly {} MB of memory. "
                "Consider merging the units "
                "or changing jac_thres".format(cur_YrA.shape[0], mem_demand)
            )
        if not warm_start:
            cur_C = None
        if cur_YrA.shape[0] > 1:
            dl_opt = inline_opt
        else:
            dl_opt = custom_delay_optimize
        # explicitly using delay (rather than gufunc) seem to promote the
        # depth-first behavior of dask
        with da.config.set(delayed_optimize=dl_opt):
            res = da.optimize(
                da.delayed(update_temporal_block)(
                    cur_YrA,
                    noise_freq=noise_freq,
                    p=p,
                    add_lag=add_lag,
                    normalize=normalize,
                    concurrent=concurrent_update,
                    use_smooth=use_smooth,
                    c_last=cur_C,
                    bseg=bseg,
                    med_wd=med_wd,
                    sparse_penal=sparse_penal,
                    max_iters=max_iters,
                    scs_fallback=scs_fallback,
                    zero_thres=zero_thres,
                )
            )[0]
        c_ls.append(darr.from_delayed(res[0], shape=cur_YrA.shape, dtype=cur_YrA.dtype))
        s_ls.append(darr.from_delayed(res[1], shape=cur_YrA.shape, dtype=cur_YrA.dtype))
        b_ls.append(darr.from_delayed(res[2], shape=cur_YrA.shape, dtype=cur_YrA.dtype))
        c0_ls.append(
            darr.from_delayed(res[3], shape=cur_YrA.shape, dtype=cur_YrA.dtype)
        )
        g_ls.append(
            darr.from_delayed(res[4], shape=(cur_YrA.shape[0], p), dtype=cur_YrA.dtype)
        )
    uids_new = np.concatenate(uid_ls)
    C_new = xr.DataArray(
        darr.concatenate(c_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new,
            "frame": YrA.coords["frame"],
        },
        name="C_new",
    )
    S_new = xr.DataArray(
        darr.concatenate(s_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new,
            "frame": YrA.coords["frame"].values,
        },
        name="S_new",
    )
    b0_new = xr.DataArray(
        darr.concatenate(b_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new,
            "frame": YrA.coords["frame"].values,
        },
        name="b0_new",
    )
    c0_new = xr.DataArray(
        darr.concatenate(c0_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new,
            "frame": YrA.coords["frame"].values,
        },
        name="c0_new",
    )
    g = xr.DataArray(
        darr.concatenate(g_ls, axis=0),
        dims=["unit_id", "lag"],
        coords={"unit_id": uids_new, "lag": np.arange(p)},
        name="g",
    )
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^update_temporal_block"])
    with da.config.set(array_optimize=arr_opt):
        da.compute(
            [
                save_minian(
                    var.chunk({"unit_id": 1}), intpath, compute=False, overwrite=True
                )
                for var in [C_new, S_new, b0_new, c0_new, g]
            ]
        )
    int_ds = open_minian(intpath, return_dict=True)
    C_new, S_new, b0_new, c0_new, g = (
        int_ds["C_new"],
        int_ds["S_new"],
        int_ds["b0_new"],
        int_ds["c0_new"],
        int_ds["g"],
    )
    mask = (S_new.sum("frame") > 0).compute()
    print("{} out of {} units dropped".format((~mask).sum().values, len(Ymask)))
    C_new, S_new, b0_new, c0_new, g = (
        C_new[mask],
        S_new[mask],
        b0_new[mask],
        c0_new[mask],
        g[mask],
    )
    sig_new = C_new + b0_new + c0_new
    sig_new = da.optimize(sig_new)[0]
    YrA_new = YrA.sel(unit_id=mask)
    if post_scal and len(sig_new) > 0:
        print("post-hoc scaling")
        scal = lstsq_vec(sig_new.data, YrA_new.data).compute().reshape((-1, 1))
        C_new, S_new, b0_new, c0_new = (
            C_new * scal,
            S_new * scal,
            b0_new * scal,
            c0_new * scal,
        )
    return C_new, S_new, b0_new, c0_new, g, mask

def custom_delay_optimize(
    dsk: dict, keys: list, fast_functions=[], inline_patterns=[], **kwargs
) -> dict:
    """
    Custom optimization functions for delayed tasks.

    By default only fusing of tasks will be carried out.

    Parameters
    ----------
    dsk : dict
        Input dask task graph.
    keys : list
        Output task keys.
    fast_functions : list, optional
        List of fast functions to be inlined. By default `[]`.
    inline_patterns : list, optional
        List of patterns of task keys to be inlined. By default `[]`.

    Returns
    -------
    dsk : dict
        Optimized dask graph.
    """
    dsk, _ = fuse(ensure_dict(dsk), rename_keys=custom_fused_keys_renamer)
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if fast_functions:
        dsk = inline_functions(
            dsk,
            [],
            fast_functions=fast_functions,
        )
    return dsk



@darr.as_gufunc(signature="(f),(f)->()", output_dtypes=float)
def lstsq_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Estimate a least-square scaling from `a` to `b` in vectorized fashion.

    Parameters
    ----------
    a : np.ndarray
        Source of the scaling.
    b : np.ndarray
        Target of the scaling.

    Returns
    -------
    scale : np.ndarray
        A scaler that scales `a` to `b`.
    """
    a = a.reshape((-1, 1))
    return np.linalg.lstsq(a, b.squeeze(), rcond=-1)[0]


def get_ar_coef(
    y: np.ndarray, sn: float, p: int, add_lag: int, pad: Optional[int] = None
) -> np.ndarray:
    """
    Estimate Autoregressive coefficients of order `p` given a timeseries `y`.

    Parameters
    ----------
    y : np.ndarray
        Input timeseries.
    sn : float
        Estimated noise level of the input `y`.
    p : int
        Order of the autoregressive process.
    add_lag : int
        Additional number of timesteps of covariance to use for the estimation.
    pad : int, optional
        Length of the output. If not `None` then the resulting coefficients will
        be zero-padded to this length. By default `None`.

    Returns
    -------
    g : np.ndarray
        The estimated AR coefficients.
    """
    if add_lag == "p":
        max_lag = p * 2
    else:
        max_lag = p + add_lag
    cov = acovf(y, fft=True)
    C_mat = toeplitz(cov[:max_lag], cov[:p]) - sn ** 2 * np.eye(max_lag, p)
    g = lstsq(C_mat, cov[1 : max_lag + 1])[0]
    if pad:
        res = np.zeros(pad)
        res[: len(g)] = g
        return res
    else:
        return g


def get_p(y):
    dif = np.append(np.diff(y), 0)
    rising = dif > 0
    prd_ris, num_ris = label(rising)
    ext_prd = np.zeros(num_ris)
    for id_prd in range(num_ris):
        prd = y[prd_ris == id_prd + 1]
        ext_prd[id_prd] = prd[-1] - prd[0]
    id_max_prd = np.argmax(ext_prd)
    return np.sum(rising[prd_ris == id_max_prd + 1])


def update_temporal_block(
    YrA: np.ndarray,
    noise_freq: float,
    p: int,
    add_lag="p",
    normalize=True,
    use_smooth=True,
    med_wd=None,
    concurrent=False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Update temporal components given residule traces of a group of cells.

    This function wraps around :func:`update_temporal_cvxpy`, but also carry out
    additional initial steps given `YrA` of a group of cells. Additional keyword
    arguments are passed through to :func:`update_temporal_cvxpy`.

    Parameters
    ----------
    YrA : np.ndarray
        Residule traces of a group of cells. Should have dimension ("unit_id",
        "frame").
    noise_freq : float
        Frequency cut-off for both the estimation of noise level and the
        optional smoothing. Specified as a fraction of sampling frequency.
    p : int
        Order of the AR process.
    add_lag : str, optional
        Additional number of timesteps in covariance to use for the estimation
        of AR coefficients. By default "p".
    normalize : bool, optional
        Whether to normalize `YrA` for each cell to unit sum. By default `True`.
    use_smooth : bool, optional
        Whether to smooth the `YrA` for the estimation of AR coefficients. By
        default `True`.
    med_wd : int, optional
        Median window used for baseline correction.
    concurrent : bool, optional
        Whether to update a group of cells as a single optimization problem. By
        default `False`.

    Returns
    -------
    c : np.ndarray
        New estimation of the calcium dynamic of the group of cells. Should have
        dimensions ("unit_id", "frame") and same shape as `YrA`.
    s : np.ndarray
        New estimation of the deconvolved spikes of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    b : np.ndarray
        New estimation of baseline fluorescence of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    c0 : np.ndarray
        New estimation of a initial calcium decay of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    g : np.ndarray
        Estimation of AR coefficient for each cell. Should have dimensions
        ("unit_id", "lag") with "lag" having length `p`.

    See Also
    -------
    update_temporal : for more explanation of parameters
    """
    vec_get_noise = np.vectorize(
        noise_fft,
        otypes=[float],
        excluded=["noise_range", "noise_method"],
        signature="(f)->()",
    )
    vec_get_p = np.vectorize(get_p, otypes=[int], signature="(f)->()")
    vec_get_ar_coef = np.vectorize(
        get_ar_coef,
        otypes=[float],
        excluded=["pad", "add_lag"],
        signature="(f),(),()->(l)",
    )
    if normalize:
        amean = YrA.sum(axis=1).mean()
        norm_factor = YrA.shape[1] / amean
        YrA *= norm_factor
    else:
        norm_factor = np.ones(YrA.shape[0])
    tn = vec_get_noise(YrA, noise_range=(noise_freq, 1))
    if use_smooth:
        YrA_ar = filt_fft_vec(YrA, noise_freq, "low")
        tn_ar = vec_get_noise(YrA_ar, noise_range=(noise_freq, 1))
    else:
        YrA_ar, tn_ar = YrA, tn
    # auto estimation of p is disabled since it's never used and makes it
    # impossible to pre-determine the shape of output
    # if p is None:
    #     p = np.clip(vec_get_p(YrA_ar), 1, None)
    pmax = np.max(p)
    g = vec_get_ar_coef(YrA_ar, tn_ar, p, pad=pmax, add_lag=add_lag)
    del YrA_ar, tn_ar
    if med_wd is not None:
        for i, cur_yra in enumerate(YrA):
            YrA[i, :] = med_baseline(cur_yra, med_wd)
    if concurrent:
        c, s, b, c0 = update_temporal_cvxpy(YrA, g, tn, **kwargs)
    else:
        res_ls = []
        for cur_yra, cur_g, cur_tn in zip(YrA, g, tn):
            res = update_temporal_cvxpy(cur_yra, cur_g, cur_tn, **kwargs)
            res_ls.append(res)
        c = np.concatenate([r[0] for r in res_ls], axis=0) / norm_factor
        s = np.concatenate([r[1] for r in res_ls], axis=0) / norm_factor
        b = np.concatenate([r[2] for r in res_ls], axis=0) / norm_factor
        c0 = np.concatenate([r[3] for r in res_ls], axis=0) / norm_factor
    return c, s, b, c0, g

def med_baseline(a: np.ndarray, wnd: int) -> np.ndarray:
    """
    Subtract baseline from a timeseries as estimated by median-filtering the
    timeseries.

    Parameters
    ----------
    a : np.ndarray
        Input timeseries.
    wnd : int
        Window size of the median filter. This parameter is passed as `size` to
        :func:`scipy.ndimage.filters.median_filter`.

    Returns
    -------
    a : np.ndarray
        Timeseries with baseline subtracted.
    """
    base = median_filter(a, size=wnd)
    a -= base
    return a.clip(0, None)

def update_temporal_cvxpy(
    y: np.ndarray, g: np.ndarray, sn: np.ndarray, A=None, bseg=None, **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the temporal update optimization problem using `cvxpy`

    Parameters
    ----------
    y : np.ndarray
        Input residule trace of one or more cells.
    g : np.ndarray
        Estimated AR coefficients of one or more cells.
    sn : np.ndarray
        Noise level of one or more cells.
    A : np.ndarray, optional
        Spatial footprint of one or more cells. Not used. By default `None`.
    bseg : np.ndarray, optional
        1d vector with length "frame" representing segments for which baseline
        should be estimated independently. By default `None`.

    Returns
    -------
    c : np.ndarray
        New estimation of the calcium dynamic of the group of cells. Should have
        dimensions ("unit_id", "frame") and same shape as `y`.
    s : np.ndarray
        New estimation of the deconvolved spikes of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    b : np.ndarray
        New estimation of baseline fluorescence of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    c0 : np.ndarray
        New estimation of a initial calcium decay of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.

    Other Parameters
    -------
    sparse_penal : float
        Sparse penalty parameter for all the cells.
    max_iters : int
        Maximum number of iterations.
    use_cons : bool, optional
        Whether to try constrained version of the problem first. By default
        `False`.
    scs_fallback : bool
        Whether to fall back to `scs` solver if the default `ecos` solver fails.
    c_last : np.ndarray, optional
        Initial estimation of calcium traces for each cell used to warm start.
    zero_thres : float
        Threshold to filter out small values in the result.

    See Also
    -------
    update_temporal : for more explanation of parameters
    """
    # spatial:
    # (d, f), (u, p), (d), (d, u)
    # (d, f), (p), (d), (d)
    # trace:
    # (u, f), (u, p), (u)
    # (f), (p), ()

    # get_parameters
    sparse_penal = kwargs.get("sparse_penal")
    max_iters = kwargs.get("max_iters")
    use_cons = kwargs.get("use_cons", False)
    scs = kwargs.get("scs_fallback")
    c_last = kwargs.get("c_last")
    zero_thres = kwargs.get("zero_thres")
    # conform variables to generalize multiple unit case
    if y.ndim < 2:
        y = y.reshape((1, -1))
    if g.ndim < 2:
        g = g.reshape((1, -1))
    sn = np.atleast_1d(sn)
    if A is not None:
        if A.ndim < 2:
            A = A.reshape((-1, 1))
    # get count of frames and units
    _T = y.shape[-1]
    _u = g.shape[0]
    if A is not None:
        _d = A.shape[0]
    # construct G matrix and decay vector per unit
    dc_vec = np.zeros((_u, _T))
    G_ls = []
    for cur_u in range(_u):
        cur_g = g[cur_u, :]
        # construct first column and row
        cur_c = np.zeros(_T)
        cur_c[0] = 1
        cur_c[1 : len(cur_g) + 1] = -cur_g
        # update G with toeplitz matrix
        G_ls.append(
            cvx.Constant(
                dia_matrix(
                    (
                        np.tile(np.concatenate(([1], -cur_g)), (_T, 1)).T,
                        -np.arange(len(cur_g) + 1),
                    ),
                    shape=(_T, _T),
                ).tocsc()
            )
        )
        # update dc_vec
        cur_gr = np.roots(cur_c)
        dc_vec[cur_u, :] = np.max(cur_gr.real) ** np.arange(_T)
    # get noise threshold
    thres_sn = sn * np.sqrt(_T)
    # construct variables
    if bseg is not None:
        nseg = int(np.max(bseg) + 1)
        b_temp = np.zeros((nseg, _T))
        for iseg in range(nseg):
            b_temp[iseg, bseg == iseg] = 1
        b_cmp = cvx.Variable((_u, nseg))
    else:
        b_temp = np.ones((1, _T))
        b_cmp = cvx.Variable((_u, 1))
    b = b_cmp @ b_temp  # baseline fluorescence per unit
    c0 = cvx.Variable(_u)  # initial fluorescence per unit
    c = cvx.Variable((_u, _T))  # calcium trace per unit
    if c_last is not None:
        c.value = c_last
        warm_start = True
    else:
        warm_start = False
    s = cvx.vstack([G_ls[u] @ c[u, :] for u in range(_u)])  # spike train per unit
    # residual noise per unit
    if A is not None:
        sig = cvx.vstack(
            [
                (A * c)[px, :] + (A * b)[px, :] + (A * cvx.diag(c0) * dc_vec)[px, :]
                for px in range(_d)
            ]
        )
        noise = y - sig
    else:
        sig = cvx.vstack([c[u, :] + b[u, :] + c0[u] * dc_vec[u, :] for u in range(_u)])
        noise = y - sig
    noise = cvx.vstack([cvx.norm(noise[i, :], 2) for i in range(noise.shape[0])])
    # construct constraints
    cons = []
    cons.append(
        b >= np.broadcast_to(np.min(y, axis=-1).reshape((-1, 1)), y.shape)
    )  # baseline larger than minimum
    cons.append(c0 >= 0)  # initial fluorescence larger than 0
    cons.append(s >= 0)  # spike train non-negativity
    # noise constraints
    cons_noise = [noise[i] <= thres_sn[i] for i in range(thres_sn.shape[0])]
    try:
        obj = cvx.Minimize(cvx.sum(cvx.norm(s, 1, axis=1)))
        prob = cvx.Problem(obj, cons + cons_noise)
        if use_cons:
            _ = prob.solve(solver="ECOS")
        if not (prob.status == "optimal" or prob.status == "optimal_inaccurate"):
            if use_cons:
                warnings.warn("constrained version of problem infeasible")
            raise ValueError
    except (ValueError, cvx.SolverError):
        lam = sn * sparse_penal
        obj = cvx.Minimize(
            cvx.sum(cvx.sum(noise, axis=1) + cvx.multiply(lam, cvx.norm(s, 1, axis=1)))
        )
        prob = cvx.Problem(obj, cons)
        try:
            _ = prob.solve(solver="ECOS", warm_start=warm_start, max_iters=max_iters)
            if prob.status in ["infeasible", "unbounded", None]:
                raise ValueError
        except (cvx.SolverError, ValueError):
            try:
                if scs:
                    _ = prob.solve(solver="SCS", max_iters=200)
                if prob.status in ["infeasible", "unbounded", None]:
                    raise ValueError
            except (cvx.SolverError, ValueError):
                warnings.warn(
                    "problem status is {}, returning zero".format(prob.status),
                    RuntimeWarning,
                )
                return [np.zeros(c.shape, dtype=float)] * 4
    if not (prob.status == "optimal"):
        warnings.warn("problem solved sub-optimally", RuntimeWarning)
    c = np.where(c.value > zero_thres, c.value, 0)
    s = np.where(s.value > zero_thres, s.value, 0)
    b = np.where(b.value > zero_thres, b.value, 0)
    c0 = c0.value.reshape((-1, 1)) * dc_vec
    c0 = np.where(c0 > zero_thres, c0, 0)
    return c, s, b, c0


def unit_merge(
    A: xr.DataArray,
    C: xr.DataArray,
    add_list: Optional[List[xr.DataArray]] = None,
    thres_corr=0.9,
    noise_freq: Optional[float] = None,
) -> Tuple[xr.DataArray, xr.DataArray, Optional[List[xr.DataArray]]]:
    """
    Merge cells given spatial footprints and temporal components

    This function merge all cells that have common pixels based on correlation
    of their temporal components. The cells to be merged will become one cell,
    with spatial and temporal components taken as mean across all the cells to
    be merged. Additionally any variables specified in `add_list` will be merged
    in the same manner. Optionally the temporal components can be smoothed
    before being used to caculate correlation. Despite the name any timeseries
    be passed as `C` and used to calculate the correlation.

    Parameters
    ----------
    A : xr.DataArray
        Spatial footprints of the cells.
    C : xr.DataArray
        Temporal component of cells.
    add_list : List[xr.DataArray], optional
        List of additional variables to be merged. By default `None`.
    thres_corr : float, optional
        The threshold of correlation. Any pair of spatially overlapping cells
        with correlation higher than this threshold will be transitively grouped
        together and merged. By default `0.9`.
    noise_freq : float, optional
        The cut-off frequency used to smooth `C` before calculation of
        correlation. If `None` then no smoothing will be done. By default
        `None`.

    Returns
    -------
    A_merge : xr.DataArray
        Merged spatial footprints of cells.
    C_merge : xr.DataArray
        Merged temporal components of cells.
    add_list : List[xr.DataArray], optional
        List of additional merged variables. Only returned if input `add_list`
        is not `None`.
    """
    print("computing spatial overlap")
    with da.config.set(
        array_optimize=darr.optimization.optimize,
        **{"optimization.fuse.subgraphs": False}
    ):
        A_sps = (A.data.map_blocks(sparse.COO) > 0).rechunk(-1).persist()
        A_inter = sparse.tril(
            darr.tensordot(
                A_sps.astype(np.float32),
                A_sps.astype(np.float32),
                axes=[(1, 2), (1, 2)],
            ).compute(),
            k=-1,
        )
    print("computing temporal correlation")
    nod_df = pd.DataFrame({"unit_id": A.coords["unit_id"].values})
    adj = adj_corr(C, A_inter, nod_df, freq=noise_freq)
    print("labeling units to be merged")
    adj = adj > thres_corr
    adj = adj + adj.T
    unit_labels = xr.apply_ufunc(
        label_connected,
        adj,
        input_core_dims=[["unit_id", "unit_id_cp"]],
        output_core_dims=[["unit_id"]],
    )
    print("merging units")
    A_merge = (
        A.assign_coords(unit_labels=("unit_id", unit_labels))
        .groupby("unit_labels")
        .mean("unit_id")
        .rename(unit_labels="unit_id")
    )
    C_merge = (
        C.assign_coords(unit_labels=("unit_id", unit_labels))
        .groupby("unit_labels")
        .mean("unit_id")
        .rename(unit_labels="unit_id")
    )
    if add_list:
        for ivar, var in enumerate(add_list):
            var_mrg = (
                var.assign_coords(unit_labels=("unit_id", unit_labels))
                .groupby("unit_labels")
                .mean("unit_id")
                .rename(unit_labels="unit_id")
            )
            add_list[ivar] = var_mrg
        return A_merge, C_merge, add_list
    else:
        return A_merge, C_merge


def label_connected(adj: np.ndarray, only_connected=False) -> np.ndarray:
    """
    Label connected components given adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix. Should be 2d symmetric matrix.
    only_connected : bool, optional
        Whether to keep only the labels of connected components. If `True`, then
        all components with only one node (isolated) will have their labels set
        to -1. Otherwise all components will have unique label. By default
        `False`.

    Returns
    -------
    labels : np.ndarray
        The labels for each components. Should have length `adj.shape[0]`.
    """
    try:
        np.fill_diagonal(adj, 0)
        adj = np.triu(adj)
        g = nx.convert_matrix.from_numpy_matrix(adj)
    except:
        g = nx.convert_matrix.from_scipy_sparse_matrix(adj)
    labels = np.zeros(adj.shape[0], dtype=np.int)
    for icomp, comp in enumerate(nx.connected_components(g)):
        comp = list(comp)
        if only_connected and len(comp) == 1:
            labels[comp] = -1
        else:
            labels[comp] = icomp
    return labels


def smooth_sig(
    sig: xr.DataArray, freq: float, method="fft", btype="low"
) -> xr.DataArray:
    """
    Filter the input timeseries with a cut-off frequency in vecorized fashion.

    Parameters
    ----------
    sig : xr.DataArray
        The input timeseries. Should have dimension "frame".
    freq : float
        The cut-off frequency.
    method : str, optional
        Method used for filtering. Either `"fft"` or `"butter"`. If `"fft"`, the
        filtering is carried out with zero-ing fft signal. If `"butter"`, the
        fiilterings carried out with :func:`scipy.signal.butter`. By default
        "fft".
    btype : str, optional
        Either `"low"` or `"high"` specify low or high pass filtering. By
        default `"low"`.

    Returns
    -------
    sig_smth : xr.DataArray
        The filtered signal. Has same shape as input `sig`.

    Raises
    ------
    NotImplementedError
        if `method` is not "fft" or "butter"
    """
    try:
        filt_func = {"fft": filt_fft, "butter": filt_butter}[method]
    except KeyError:
        raise NotImplementedError(method)
    sig_smth = xr.apply_ufunc(
        filt_func,
        sig,
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
        kwargs={"btype": btype, "freq": freq},
        dask="parallelized",
        output_dtypes=[sig.dtype],
    )
    return sig_smth


def filt_fft(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Filter 1d timeseries by zero-ing bands in the fft signal.

    Parameters
    ----------
    x : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency.
    btype : str
        Either `"low"` or `"high"` specify low or high pass filtering.

    Returns
    -------
    x_filt : np.ndarray
        Filtered timeseries.
    """
    _T = len(x)
    if btype == "low":
        zero_range = slice(int(freq * _T), None)
    elif btype == "high":
        zero_range = slice(None, int(freq * _T))
    xfft = numpy_fft.rfft(x)
    xfft[zero_range] = 0
    return numpy_fft.irfft(xfft, len(x))


def filt_butter(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Filter 1d timeseries with Butterworth filter using
    :func:`scipy.signal.butter`.

    Parameters
    ----------
    x : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency.
    btype : str
        Either "low" or "high" specify low or high pass filtering.

    Returns
    -------
    x_filt : np.ndarray
        Filtered timeseries.
    """
    but_b, but_a = butter(2, freq * 2, btype=btype, analog=False)
    return lfilter(but_b, but_a, x)


def filt_fft_vec(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Vectorized wrapper of :func:`filt_fft`.

    Parameters
    ----------
    x : np.ndarray
        Input timeseries. Should have 2 dimensions, and the filtering will be
        applied along the last dimension.
    freq : float
        Cut-off frequency.
    btype : str
        Either `"low"` or `"high"` specify low or high pass filtering.

    Returns
    -------
    x_filt : np.ndarray
        Filtered timeseries
    """
    for ix, xx in enumerate(x):
        x[ix, :] = filt_fft(xx, freq, btype)
    return x


def compute_AtC(A: xr.DataArray, C: xr.DataArray) -> xr.DataArray:
    """
    Compute the outer product of spatial and temporal components.

    This funtion computes the outer product of spatial and temporal components.
    The result is a 3d array representing the movie data as estimated by the
    spatial and temporal components.

    Parameters
    ----------
    A : xr.DataArray
        Spatial footprints of cells. Should have dimensions ("unit_id",
        "height", "width").
    C : xr.DataArray
        Temporal components of cells. Should have dimensions "frame" and
        "unit_id".

    Returns
    -------
    AtC : xr.DataArray
        The outer product representing estimated movie data. Has dimensions
        ("frame", "height", "width").
    """
    fm, h, w = (
        C.coords["frame"].values,
        A.coords["height"].values,
        A.coords["width"].values,
    )
    A = darr.from_array(
        A.data.map_blocks(sparse.COO, dtype=A.dtype).compute(), chunks=-1
    )
    C = C.transpose("frame", "unit_id").data.map_blocks(sparse.COO, dtype=C.dtype)
    AtC = darr.tensordot(C, A, axes=(1, 0)).map_blocks(
        lambda a: a.todense(), dtype=A.dtype
    )
    arr_opt = fct.partial(
        custom_arr_optimize, rename_dict={"tensordot": "tensordot_restricted"}
    )
    with da.config.set(array_optimize=arr_opt):
        AtC = da.optimize(AtC)[0]
    return xr.DataArray(
        AtC,
        dims=["frame", "height", "width"],
        coords={"frame": fm, "height": h, "width": w},
    )


def graph_optimize_corr(
    varr: xr.DataArray,
    G: nx.Graph,
    freq: float,
    idx_dims=["height", "width"],
    chunk=600,
    step_size=50,
) -> pd.DataFrame:
    """
    Compute correlation in an optimized fashion given a computation graph.

    This function carry out out-of-core computation of large correaltion matrix.
    It takes in a computaion graph whose node represent timeseries and edges
    represent the desired pairwise correlation to be computed. The actual
    timeseries are stored in `varr` and indexed with node attributes. The
    function can carry out smoothing of timeseries before computation of
    correlation. To minimize re-computation of smoothing for each pixel, the
    graph is first partitioned using a minial-cut algorithm. Then the
    computation is performed in chunks with size `chunk`, with nodes from the
    same partition being in the same chunk as much as possible.

    Parameters
    ----------
    varr : xr.DataArray
        Input timeseries. Should have "frame" dimension in addition to those
        specified in `idx_dims`.
    G : nx.Graph
        Graph representing computation to be carried out. Should be undirected
        and un-weighted. Each node should have unique attributes with keys
        specified in `idx_dims`, which will be used to index the timeseries in
        `varr`. Each edge represent a desired correlation.
    freq : float
        Cut-off frequency for the optional smoothing. If `None` then no
        smoothing will be done.
    idx_dims : list, optional
        The dimension used to index the timeseries in `varr`. By default
        `["height", "width"]`.
    chunk : int, optional
        Chunk size of each computation. By default `600`.
    step_size : int, optional
        Step size to iterate through all edges. If too small then the iteration
        will take a long time. If too large then the variances in the actual
        chunksize of computation will be large. By default `50`.

    Returns
    -------
    eg_df : pd.DataFrame
        Dataframe representation of edge list. Has column "source" and "target"
        representing the node index of the edge (correlation), and column "corr"
        with computed value of correlation.
    """
    # a heuristic to make number of partitions scale with nodes
    # following is not compatible with windows
    # n_cuts, membership = metis.part_graph(
    # max(int(np.ceil(G.number_of_nodes() / chunk)), 1), adjacency=adj_list(G))

    #### replacement code for windows:
    # note needs python-louvain & community


    # Perform community detection
    partition = community.best_partition(G)

    # Convert partition dictionary to membership list
    membership = np.array([partition[node] for node in G.nodes()])

    # Calculate the number of cuts
    # num_cuts = max(int(np.ceil(G.number_of_nodes() / chunk)), 1)

    # Generate the adjacency matrix (this gives memory error)
    # adj_matrix = nx.to_numpy_array(G)

    # Perform spectral clustering
    # sc = SpectralClustering(n_clusters=num_cuts, affinity='precomputed')
    # membership = sc.fit_predict(adj_matrix)

    # Assign number of cuts
    # n_cuts = num_cuts

    nx.set_node_attributes(
        G, {k: {"part": v} for k, v in zip(sorted(G.nodes), membership)}
    )
    eg_df = nx.to_pandas_edgelist(G)
    part_map = nx.get_node_attributes(G, "part")
    eg_df["part_src"] = eg_df["source"].map(part_map)
    eg_df["part_tgt"] = eg_df["target"].map(part_map)
    eg_df["part_diff"] = (eg_df["part_src"] - eg_df["part_tgt"]).astype(bool)
    corr_ls = []
    idx_ls = []
    npxs = []
    egd_same, egd_diff = eg_df[~eg_df["part_diff"]], eg_df[eg_df["part_diff"]]
    idx_dict = {d: nx.get_node_attributes(G, d) for d in idx_dims}

    def construct_comput(edf, pxs):
        px_map = {k: v for v, k in enumerate(pxs)}
        ridx = edf["source"].map(px_map).values
        cidx = edf["target"].map(px_map).values
        idx_arr = {
            d: xr.DataArray([dd[p] for p in pxs], dims="pixels")
            for d, dd in idx_dict.items()
        }
        vsub = varr.sel(**idx_arr).data
        if len(idx_arr) > 1:  # vectorized indexing
            vsub = vsub.T
        else:
            vsub = vsub.rechunk(-1)
        with da.config.set(**{"optimization.fuse.ave-width": vsub.shape[0]}):
            return da.optimize(smooth_corr(vsub, ridx, cidx, freq=freq))[0]

    for _, eg_sub in egd_same.groupby("part_src"):
        pixels = list(set(eg_sub["source"]) | set(eg_sub["target"]))
        corr_ls.append(construct_comput(eg_sub, pixels))
        idx_ls.append(eg_sub.index)
        npxs.append(len(pixels))
    pixels = set()
    eg_ls = []
    grp = np.arange(len(egd_diff)) // step_size
    for igrp, eg_sub in egd_diff.sort_values("source").groupby(grp):
        pixels = pixels | set(eg_sub["source"]) | set(eg_sub["target"])
        eg_ls.append(eg_sub)
        if (len(pixels) > chunk - step_size / 2) or igrp == max(grp):
            pixels = list(pixels)
            edf = pd.concat(eg_ls)
            corr_ls.append(construct_comput(edf, pixels))
            idx_ls.append(edf.index)
            npxs.append(len(pixels))
            pixels = set()
            eg_ls = []
    print("pixel recompute ratio: {}".format(sum(npxs) / G.number_of_nodes()))
    print("computing correlations")
    corr_ls = da.compute(corr_ls)[0]
    corr = pd.Series(np.concatenate(corr_ls), index=np.concatenate(idx_ls), name="corr")
    eg_df["corr"] = corr
    return eg_df


def adj_corr(
    varr: xr.DataArray, adj: np.ndarray, nod_df: pd.DataFrame, freq: float
    ) -> scipy.sparse.csr_matrix:
    """
    Compute correlation in an optimized fashion given an adjacency matrix and
    node attributes.

    Wraps around :func:`graph_optimize_corr` and construct computation graph
    from `adj` and `nod_df`. Also convert the result into a sparse matrix with
    same shape as `adj`.

    Parameters
    ----------
    varr : xr.DataArray
        Input time series. Should have "frame" dimension in addition to column
        names of `nod_df`.
    adj : np.ndarray
        Adjacency matrix.
    nod_df : pd.DataFrame
        Dataframe containing node attributes. Should have length `adj.shape[0]`
        and only contain columns relevant to index the time series.
    freq : float
        Cut-off frequency for the optional smoothing. If `None` then no
        smoothing will be done.

    Returns
    -------
    adj_corr : scipy.sparse.csr_matrix
        Sparse matrix of the same shape as `adj` but with values corresponding
        the computed correlation.
    """
    G = nx.Graph()
    G.add_nodes_from([(i, d) for i, d in enumerate(nod_df.to_dict("records"))])
    G.add_edges_from([(s, t) for s, t in zip(*adj.nonzero())])
    corr_df = graph_optimize_corr(varr, G, freq, idx_dims=nod_df.columns)
    return scipy.sparse.csr_matrix(
        (corr_df["corr"], (corr_df["source"], corr_df["target"])), shape=adj.shape
    )


def adj_list(G: nx.Graph) -> List[np.ndarray]:
    """
    Generate adjacency list representation from graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    adj_ls : List[np.ndarray]
        The adjacency list representation of graph.
    """
    gdict = nx.to_dict_of_dicts(G)
    return [np.array(list(gdict[k].keys())) for k in sorted(gdict.keys())]


@darr.as_gufunc(signature="(p,f),(i),(i)->(i)", output_dtypes=[float])
def smooth_corr(
    X: np.ndarray, ridx: np.ndarray, cidx: np.ndarray, freq: float
) -> np.ndarray:
    """
    Wraps around :func:`filt_fft_vec` and :func:`idx_corr` to carry out both
    smoothing and computation of partial correlation.

    Parameters
    ----------
    X : np.ndarray
        Input time series.
    ridx : np.ndarray
        Row index of the resulting correlation.
    cidx : np.ndarray
        Column index of the resulting correlation.
    freq : float
        Cut-off frequency for the smoothing.

    Returns
    -------
    corr : np.ndarray
        Resulting partial correlation.
    """
    if freq:
        X = filt_fft_vec(X, freq, "low")
    return idx_corr(X, ridx, cidx)


@nb.jit(nopython=True, nogil=True, cache=True)
def idx_corr(X: np.ndarray, ridx: np.ndarray, cidx: np.ndarray) -> np.ndarray:
    """
    Compute partial pairwise correlation based on index.

    This function compute a subset of a pairwise correlation matrix. The
    correlation to be computed are specified by two vectors `ridx` and `cidx` of
    same length, representing the row and column index of the full correlation
    matrix. The function use them to index the timeseries matrix `X` and compute
    only the requested correlations. The result is returned flattened.

    Parameters
    ----------
    X : np.ndarray
        Input time series. Should have 2 dimensions, where the last dimension
        should be the time dimension.
    ridx : np.ndarray
        Row index of the correlation.
    cidx : np.ndarray
        Column index of the correlation.

    Returns
    -------
    res : np.ndarray
        Flattened resulting correlations. Has same shape as `ridx` or `cidx`.
    """
    res = np.zeros(ridx.shape[0])
    std = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        X[i, :] -= X[i, :].mean()
        std[i] = np.sqrt((X[i, :] ** 2).sum())
    for i, (r, c) in enumerate(zip(ridx, cidx)):
        cur_std = std[r] * std[c]
        if cur_std > 0:
            res[i] = (X[r, :] * X[c, :]).sum() / cur_std
        else:
            res[i] = 0
    return res


def update_background(
    Y: xr.DataArray, A: xr.DataArray, C: xr.DataArray, b: xr.DataArray = None
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Update background terms given spatial and temporal components of cells.

    A movie representation (with dimensions "height" "width" and "frame") of
    estimated cell activities are computed as the product between the spatial
    components matrix and the temporal components matrix of cells over the
    "unit_id" dimension. Then the residule movie is computed by subtracting the
    estimated cell activity movie from the input movie. Then the spatial
    footprint of background `b` is the mean of the residule movie over "frame"
    dimension, and the temporal component of background `f` is the least-square
    solution between the residule movie and the spatial footprint `b`.

    Parameters
    ----------
    Y : xr.DataArray
        Input movie data. Should have dimensions ("frame", "height", "width").
    A : xr.DataArray
        Estimation of spatial footprints of cells. Should have dimensions
        ("unit_id", "height", "width").
    C : xr.DataArray
        Estimation of temporal activities of cells. Should have dimensions
        ("unit_id", "frame").
    b : xr.DataArray, optional
        Previous estimation of spatial footprint of background. If provided it
        will be returned as-is, and only temporal activity of background will be
        updated

    Returns
    -------
    b_new : xr.DataArray
        New estimation of the spatial footprint of background. Has
        dimensions ("height", "width").
    f_new : xr.DataArray
        New estimation of the temporal activity of background. Has dimension
        "frame".
    """
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    AtC = compute_AtC(A, C)
    Yb = (Y - AtC).clip(0)
    Yb = save_minian(Yb.rename("Yb"), intpath, overwrite=True)
    if b is None:
        b_new = Yb.mean("frame").persist()
    else:
        b_new = b.persist()
    b_stk = (
        b_new.stack(spatial=["height", "width"])
        .transpose("spatial")
        .expand_dims("dummy", axis=-1)
        .chunk(-1)
    )
    Yb_stk = Yb.stack(spatial=["height", "width"]).transpose("spatial", "frame")
    f_new = darr.linalg.lstsq(b_stk.data, Yb_stk.data)[0]
    f_new = xr.DataArray(
        f_new.squeeze(), dims=["frame"], coords={"frame": Yb.coords["frame"]}
    ).persist()
    return b_new, f_new

@darr.as_gufunc(signature="(m,n),(m)->(n)", output_dtypes=float)

def sps_lstsq(a: scipy.sparse.csc_matrix, b: np.ndarray, **kwargs):
    out = np.zeros((b.shape[0], a.shape[1]))
    with da.config.set(**{'array.slicing.split_large_chunks': False}): #found this was needed to stop errors
        for i in range(b.shape[0]):
            out[i, :] = lsqr(a, b[i, :].squeeze(), **kwargs)[0]
    return out




