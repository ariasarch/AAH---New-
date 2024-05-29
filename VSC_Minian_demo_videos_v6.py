#!/usr/bin/env python
# coding: utf-8

# Step 1: Import Necessary Modules 

import time
import os
import sys
import csv
import itertools as itt
import holoviews as hv
from bokeh.plotting import show
import numpy as np
import multiprocessing as mp
from dask.distributed import Client, LocalCluster
from holoviews.operation.datashader import datashade, regrid
from holoviews.util import Dynamic
from IPython.display import display
import moviepy.editor as mpy

print("\033[1;32mStep 1 Complete\033[0m")

# Step 2: Set up Initial Basic Parameters

minian_path = "."
intpath = "./minian_intermediate"
subset = dict(frame=slice(0, None))
subset_mc = None
output_size = 100
n_workers = int(os.getenv("MINIAN_NWORKERS", 4))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MINIAN_INTERMEDIATE"] = intpath

print("\033[1;32mStep 2 Complete\033[0m")

# Step 3: Import related functions from Minian

sys.path.append(minian_path)

from minian.cnmf import (
    compute_AtC, compute_trace, get_noise_fft, smooth_sig,
    unit_merge, update_spatial, update_temporal, update_background
)

from minian.initialization import (
    initA, initC, ks_refine, pnr_refine,
    seeds_init, seeds_merge
)

from minian.motion_correction import (
    apply_transform, estimate_motion, xr
)

from minian.preprocessing import (
    denoise, remove_background
)

from minian.utilities import (
    TaskAnnotation, get_optimal_chk, load_videos,
    open_minian, save_minian
)

from minian.visualization import (
    CNMFViewer, VArrayViewer, generate_videos, 
    visualize_gmm_fit, visualize_preprocess, 
    visualize_spatial_update, visualize_temporal_update, 
    write_video
)

print("\033[1;32mStep 3 Complete\033[0m")

# Step 4: Begin Cluster

mp.freeze_support()

if os.name == 'posix': 
    mp.set_start_method('fork', force=True)
else:
    mp.set_start_method('spawn', force=True)

def initialize_dask():
    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit="4GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address=":8787",
    )
    client = Client(cluster)
    return client, cluster

def run_minian(param_path, vid_path, timestamp_file):

    client, cluster = initialize_dask()
    dpath = vid_path
    minian_ds_path = os.path.join(dpath, "minian")
    dpath = os.path.abspath(dpath)
    hv.extension("bokeh", width=100)

    print("Dashboard is available at:", client.dashboard_link)
    print("\033[1;32mStep 4 Complete\033[0m")

    def convert_value(value):
        if value.isnumeric():
            return int(value)
        elif value.replace('.', '', 1).isdigit():
            return float(value)
        elif value == "TRUE":
            return True
        elif value == "FALSE":
            return False
        elif value == "None":
            return None
        elif value == "Nan":
            print("Not a number")
            return None
        elif value[0] == "(":
            substrings = value[1:-1].split(", ")
            first = convert_value(substrings[0])
            second = convert_value(substrings[1])
            return (first, second)
        return value

    def populate_parameters(headers, row):
        params = {}
        for i, header in enumerate(headers):
            if i % 2 == 0:
                param_key = row[i]
                if param_key and param_key != "Nan":
                    param_value = row[i + 1]
                    params[param_key] = convert_value(param_value)
        return params

    def read_csv_to_params(file_path):
        params_dict = {}
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            for row in reader:
                function_name = row[0]
                params = populate_parameters(headers[1:], row[1:])
                params_dict[function_name] = params
        return params_dict

    param_dict = read_csv_to_params(param_path)
    print(param_dict)

    param_save_minian = {
        "dpath": minian_ds_path,
        "meta_dict": dict(session=-1, animal=-2),
        "overwrite": True,
    }

    param_load_videos = {
        "pattern": ".*\.(avi|mp4|mkv)$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    }

    param_remove_glow = param_dict['remove_glow']
    param_denoise = param_dict['denoise']
    param_background_removal = param_dict['background_removal']
    param_estimate_motion = param_dict['estimate_motion']
    param_seeds_init = param_dict['seeds_init']
    param_pnr_refine = param_dict['pnr_refine']
    param_ks_refine = param_dict['ks_refine']
    param_seeds_merge = param_dict['seeds_merge']
    param_initialize = param_dict['initialize']
    param_init_merge = param_dict['init_merge']
    param_get_noise = param_dict['get_noise']
    param_first_spatial = param_dict['first_spatial']
    param_first_temporal = param_dict['first_temporal']
    param_first_merge = param_dict['first_merge']
    param_second_spatial = param_dict['second_spatial']
    param_second_temporal = param_dict['second_temporal']

    print("\033[1;32mStep 5 Complete\033[0m")

    # Step 6: Preprocessing 
    start_time = time.time()

    varr = load_videos(dpath, **param_load_videos)
    chk, _ = get_optimal_chk(varr, dtype=float)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )
    hv.output(size=output_size)
    elapsed_time = time.time() - start_time     
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 6 Complete\033[0m")

    # Step 7: Perform Glow Removal 

    start_time = time.time()

    if param_remove_glow["deglow"]:
        varr_ref = varr.sel(subset) 
        varr_min = varr_ref.min("frame").compute()
        varr_ref = varr_ref - varr_min
    else:
        varr_ref = varr
    
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 7 Complete\033[0m")

    # Step 8: Perform Denoising

    start_time = time.time()

    varr_ref = denoise(varr_ref, **param_denoise)
    varr_ref = remove_background(varr_ref, **param_background_removal)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)
    
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 8 Complete\033[0m")

    # Step 9: Perform Motion Correction

    start_time = time.time()

    motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)
    motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian)
    Y = apply_transform(varr_ref, motion, fill=0)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True, chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})
    
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 9 Complete\033[0m")

    # Step 10: Split Video Based on Timestamps

    def read_timestamps(file):
        with open(file, 'r') as f:
            timestamps = [line.strip().split() for line in f.readlines()]
        return [(float(start), float(end)) for start, end in timestamps]

    timestamps = read_timestamps(timestamp_file)
    video = mpy.VideoFileClip(os.path.join(dpath, "your_video_file.mp4"))

    for i, (start, end) in enumerate(timestamps):
        subclip = video.subclip(start, end)
        subclip.write_videofile(os.path.join(dpath, f"subclip_{i+1}.mp4"), codec='libx264')

    print("\033[1;32mStep 10 Complete\033[0m")

    # Steps 11-20: Run the Remaining Steps Using Dask
    def process_step(step_func, *args, **kwargs):
        result = step_func(*args, **kwargs)
        result_future = save_minian(result, intpath, overwrite=True)
        wait(result_future)
        return result
    
    # Define a function for each remaining step to execute them with Dask

    def seeds_init_wrapper:
        param = param_dict['seeds_init']
        return seeds_init(Y_fm_chk, **param)

    def pnr_refine_wrapper:
        param = param_dict['pnr_refine']
        return pnr_refine(Y_hw_chk, seeds, **param)

    def ks_refine_wrapper:
        param = param_dict['ks_refine']
        return ks_refine(Y_hw_chk, seeds, **param)

    def step_14():
        param = param_dict['seeds_merge']
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        return seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)
        #return seeds_merge(Y_hw_chk, seeds, **param)

    def step_15():
        param = param_dict['initialize']
        return initA(Y_hw_chk, seeds, **param)

    def step_16():
        param = param_dict['init_merge']
        return initC(Y_hw_chk, A, **param)

    def step_17():
        param = param_dict['get_noise']
        return get_noise_fft(Y_hw_chk, C, **param)

    def step_18():
        param = param_dict['first_spatial']
        return update_spatial(Y_hw_chk, A, C, **param)

    def step_19():
        param = param_dict['first_temporal']
        return update_temporal(Y_hw_chk, A, C, **param)

    def step_20():
        param = param_dict['first_merge']
        return unit_merge(A, C, **param)

    def step_21():
        param = param_dict['second_spatial']
        return update_spatial(Y_hw_chk, A, C, **param)

    def step_22():
        param = param_dict['second_temporal']
        return update_temporal(Y_hw_chk, A, C, **param)

    tasks = [step_11, step_12, step_13, step_14, step_15, step_16, step_17, step_18, step_19, step_20, step_21, step_22]

    for i, task in enumerate(tasks):
        start_time = time.time()
        result = client.submit(task).result()
        elapsed_time = time.time() - start_time
        print(f"Step {i+11} Complete in {elapsed_time:.2f} seconds")

    print("\033[1;32mAll Steps Complete\033[0m")

if __name__ == "__main__":
    param_path = "parameters.csv"
    vid_path = "video_folder"
    timestamp_file = "timestamps.txt"
    run_minian(param_path, vid_path, timestamp_file)