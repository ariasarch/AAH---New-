#!/usr/bin/env python
# coding: utf-8

# %%

# Step 1: Import Necessary Modules 

import time
start_time_total = time.time()

import csv
import os # Import portable OS-dependent functionality 
import sys # Provides Direct Interaction with Python Variables 
import itertools as itt # For Efficient Looping
import holoviews as hv # Used for Visualization 
from bokeh.plotting import show # For in-line plotting
import numpy as np # For numerical computations
import multiprocessing as mp # Allow for multiproccessing 
from dask.distributed import Client, LocalCluster # Parallel Computing Library - for faster computation
from holoviews.operation.datashader import datashade, regrid # Dynamically Shade Large Datasets
from holoviews.util import Dynamic # Create Dynamic Objects
from IPython.display import display # Display Objects

print("\033[1;32mStep 1 Complete\033[0m")

# %%

# Step 2: Set up Initial Basic Parameters

# Define variables
minian_path = "." # Define the path where the Minan Module is located, a "." means where this notebook is running
intpath = "./minian_intermediate" # Define path for intermediate results 
subset = dict(frame=slice(0, None)) # Define subset of frames to process (here, all frames are included)
subset_mc = None # Motion correction subset (initialized to None)
output_size = 100 # Output size, possibly used for visualization
n_workers = int(os.getenv("MINIAN_NWORKERS", 4)) # Number of workers for parallel computation (default set to 4), this could just be replaced by n_workers = 4

# Set the number of threads for various libraries to 1
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MINIAN_INTERMEDIATE"] = intpath

print("\033[1;32mStep 2 Complete\033[0m")

# %%

# Step 3: Import related functions from Minian

sys.path.append(minian_path) # Append the Minian path to the system path to ensure Python can locate the Minian module

# Import functions related to the CNMF process from Minian
from minian.cnmf import (
    compute_AtC,             # Function computing product of spatial & temporal matrices
    compute_trace,           # Function to compute a trace (likely a temporal trace of activity)
    get_noise_fft,           # Function to estimate noise using FFT
    smooth_sig,              # Function to smooth signals
    unit_merge,              # Function to merge units (neuronal sources)
    update_spatial,          # Function to update spatial footprints of sources
    update_temporal,         # Function to update temporal activity of sources
    update_background,       # Function to update the estimated background
)

# Import functions related to initialization of sources and seeds in Minian
from minian.initialization import (
    initA,                   # Initialize spatial footprints
    initC,                   # Initialize temporal activity
    ks_refine,               # Refine initial estimates using K-S test
    pnr_refine,              # Refine initial estimates using Peak-to-Noise Ratio
    seeds_init,              # Initialize seeds or starting points for source detection
    seeds_merge,             # Merge seeds that are close or similar
)

# Import functions related to motion correction in Minian
from minian.motion_correction import (
    apply_transform,         # Apply estimated motion transformations to videos
    estimate_motion,         # Estimate motion in the videos
    xr,                      # 
)

# Import pre-processing functions from Minian
from minian.preprocessing import (
    denoise,                 # Denoise video data
    remove_background,       # Remove background from video data
)

# Import utility functions from Minian
from minian.utilities import (
    TaskAnnotation,          # Likely a tool to annotate tasks or steps
    get_optimal_chk,         # Function to determine optimal chunk sizes for processing
    load_videos,             # Function to load video datasets
    open_minian,             # Open a Minian dataset
    save_minian,             # Save a Minian dataset
)

# Import visualization tools from Minian
from minian.visualization import (
    CNMFViewer,              # Viewer tool to inspect results of CNMF
    VArrayViewer,            # Viewer tool to inspect array-like data
    generate_videos,         # Function to generate videos, possibly of processed data
    visualize_gmm_fit,       # Visualize fit of Gaussian Mixture Model
    visualize_preprocess,    # Visualize results of preprocessing
    visualize_spatial_update,# Visualize updates to spatial footprints
    visualize_temporal_update,# Visualize updates to temporal activity
    write_video,             # Function to write videos, possibly for saving processed data
)

print("\033[1;32mStep 3 Complete\033[0m")

# %%

# Step 4: Begin Cluster

# Ensure that all spawned processes are frozen and behaves as expected
mp.freeze_support()

# Allow for cross compatabilty 
if os.name == 'posix': 
    mp.set_start_method('fork', force=True)  # Unix-like operating system
else:
    mp.set_start_method('spawn', force=True)  # default for Windows

# Initialize a local Dask cluster and return a client to interact with it
def initialize_dask():

    from dask.distributed import LocalCluster, Client

    # Set up a local cluster with custom configuration
    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit="4GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address=":8787",
    )

    # Add custom task annotations to the cluster's scheduler
    # annt_plugin = TaskAnnotation()
    # cluster.scheduler.add_plugin(annt_plugin)

    # Create a client to interact with the local cluster
    client = Client(cluster)

    return client, cluster

def process_chunk(Y, i, chk, param_save_minian, param_dict):
    Y_fm_chk = save_minian(Y.astype(float).rename(f"Y_fm_chk_{i}"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename(f"Y_hw_chk_{i}"), intpath, overwrite=True, chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    param_seeds_init = param_dict['seeds_init']

    param_pnr_refine = param_dict['pnr_refine']
    param_ks_refine = param_dict['ks_refine']
    param_seeds_merge = param_dict['seeds_merge']
    param_initialize = param_dict['initialize']
    param_init_merge = param_dict['init_merge']

    # CNMF (Constrained Non-negative Matrix Factorization) Parameters
    param_get_noise = param_dict['get_noise']

    # Parameters for the first spatial update
    param_first_spatial = param_dict['first_spatial']

    # Parameters for the first temporal update
    param_first_temporal = param_dict['first_temporal']

    param_first_merge = param_dict['first_merge']

    # Parameters for the second spatial update
    param_second_spatial = param_dict['second_spatial']

    # Parameters for the second temporal update
    param_second_temporal = param_dict['second_temporal']

    # Step 10: Generating Single ROIs 
    start_time = time.time() 
    # Save Max Projection 
    max_proj = save_minian(Y_fm_chk.max("frame").rename(f"max_proj_{i}"), **param_save_minian).compute()
    #Y_hw_chk=Y_hw_chk.chunk({"frame": -1})
    # Set Seed Initializtion
    seeds = seeds_init(Y_fm_chk, **param_seeds_init)

    # Update via pnr_refine
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine) # thres = auto assumes a Guassian Mixture Model

    if gmm:
        display(visualize_gmm_fit(pnr, gmm, 100))
    else:
        print("nothing to show")

    # Refine via a Kolmogorov-Smirnov test
    seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 10 Complete\033[0m")

    # %%

    # Step 11: Initializing CNMF

    start_time = time.time() 

    print("before seeds_final update")

    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)

    print("before seeds_final2 update")
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)

    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)

    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})

    A, C = unit_merge(A_init, C_init, **param_init_merge)
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"),intpath,overwrite=True,chunks={"unit_id": -1, "frame": chk["frame"]})

    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 11 Complete\033[0m")

    # %%

    # Step 12: Estimating Spatial Noise for CNMF

    start_time = time.time() 

    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_first_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 12 Complete\033[0m")

    # %%

    # Step 13: Estimating Temporal Activity for CNMF

    start_time = time.time() 

    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_second_temporal)

    # Save Units
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)

    # Save Merged Units
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]},)
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 13 Complete\033[0m")

    # %%

    # Step 14: Performing Second Spatial Update for CNMF

    start_time = time.time() 

    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_second_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)

    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    # Save Results
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 14 Complete\033[0m")

    # %%

    # Step 15: Performing Second Temporal Update for CNMF

    start_time = time.time() 

    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_second_temporal)

    # Save Results 
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # Visualization 
    generate_videos(varr.sel(subset), Y_fm_chk, A=A, C=C_chk, vpath=dpath)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 15 Complete\033[0m")

    # %%

    # Step 16: Saving Results

    start_time = time.time()

    A = save_minian(A.rename("A"), **param_save_minian)
    C = save_minian(C.rename("C"), **param_save_minian)
    S = save_minian(S.rename("S"), **param_save_minian)
    c0 = save_minian(c0.rename("c0"), **param_save_minian)
    b0 = save_minian(b0.rename("b0"), **param_save_minian)
    b = save_minian(b.rename("b"), **param_save_minian)
    f = save_minian(f.rename("f"), **param_save_minian)

    client.close()
    cluster.close()

    print(A)
    print(C)
# Runs the minian pipeline given parameter path and video path
def run_minian(param_path, vid_path):

    client, cluster = initialize_dask()
    dpath = vid_path
    minian_ds_path = os.path.join(dpath, "minian") # Define the path where the Demo movies are stored
    dpath = os.path.abspath(dpath)
    hv.extension("bokeh", width=100)

    print("Dashboard is available at:", client.dashboard_link)
    print("\033[1;32mStep 4 Complete\033[0m")


    # Step 5: Get Function Parameters
    def convert_value(value):
        if value.isnumeric():
            value = int(value)
        elif value.replace('.', '', 1).isdigit(): # bug prone area
            value = float(value)
        elif value == "TRUE":
            value = True
        elif value == "FALSE":
            value = False
        elif value == "None":
            value = None
        elif value == "Nan":
            print("Not a number")
        elif value[0] == "(": # function assumes the following format (val, val)
            substrings = value[1:-1].split(", ")
            first = substrings[0]
            first = convert_value(first)
            second = substrings[1]
            second = convert_value(second)
            value = (first, second)
        return value

    # populates parameters from CSV rows
    def populate_parameters(headers, row):
        params = {}
        for i, header in enumerate(headers):
            if i % 2 == 0:
                param_key = row[i]
                if (param_key and param_key != "Nan"):
                    param_value = row[i + 1]
                    params[param_key] = convert_value(param_value)
        return params

    # reads the csv file and populates dictionary
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

    # Parameters for saving Minian dataset
    param_save_minian = {
        "dpath": minian_ds_path,
        "meta_dict": dict(session=-1, animal=-2), #rat1_session2.avi for example
        "overwrite": True,
    }

    # Parameters for Preprocessing to load video data
    param_load_videos = {
        "pattern": "msCam[0-9]+\.avi$", # Match file names that start with msCam by one or more digits with files that end in .avi
        "dtype": np.uint8, # 8-bit integers (0 - 255)
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    }

    # Parameters to denoise the video and for background removal
    param_remove_glow = param_dict['remove_glow']
    param_denoise = param_dict['denoise']
    param_background_removal = param_dict['background_removal']

    # Motion Correction Parameters
    subset_mc = None # Subset for motion correction
    param_estimate_motion = param_dict['estimate_motion']

    # Initialization Parameters
    param_seeds_init = param_dict['seeds_init']

    param_pnr_refine = param_dict['pnr_refine']
    param_ks_refine = param_dict['ks_refine']
    param_seeds_merge = param_dict['seeds_merge']
    param_initialize = param_dict['initialize']
    param_init_merge = param_dict['init_merge']

    # CNMF (Constrained Non-negative Matrix Factorization) Parameters
    param_get_noise = param_dict['get_noise']

    # Parameters for the first spatial update
    param_first_spatial = param_dict['first_spatial']

    # Parameters for the first temporal update
    param_first_temporal = param_dict['first_temporal']

    param_first_merge = param_dict['first_merge']

    # Parameters for the second spatial update
    param_second_spatial = param_dict['second_spatial']

    # Parameters for the second temporal update
    param_second_temporal = param_dict['second_temporal']
    print("\033[1;32mStep 5 Complete\033[0m")

    # %%

    # Step 6: Preprocessing 
    start_time = time.time() 

    varr = load_videos(dpath, **param_load_videos)
    chk, _ = get_optimal_chk(varr, dtype=float) # Estimate optimal chunk size for computations

    # Re-chunk the video array (change its internal block division for efficient computation)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )

    # Visualization of Raw Data and optionally set roi for motion correction
    hv.output(size=output_size)
    elapsed_time = time.time() - start_time     
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 6 Complete\033[0m")

    # %%

    # Step 7: Perform Glow Removal 

    start_time = time.time() 

    if (param_remove_glow["deglow"]):
        varr_ref = varr.sel(subset) 
        varr_min = varr_ref.min("frame").compute() 
        varr_ref = varr_ref - varr_min 
    else:
        varr_ref = varr

    elapsed_time = time.time() - start_time     
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 7 Complete\033[0m")

    # %%

    # Step 8: Perform Denoising

    start_time = time.time()

    varr_ref = denoise(varr_ref, **param_denoise)

    varr_ref = remove_background(varr_ref, **param_background_removal)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)

    elapsed_time = time.time() - start_time     
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 8 Complete\033[0m")

    # %%

    # Step 9: Perform Motion Correction   

    start_time = time.time() 

    # Estimate Motion
    motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)
    motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian)

    # Update our transformation as Y_hw_chk
    Y = apply_transform(varr_ref, motion, fill=0) 
    # Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    # Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True, chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 9 Complete\033[0m")

    # %%

    # here split the video into smaller videos
    # Define the number of frames per smaller video
    frames_per_split = 100 
    chunk_futures = []

    def split_and_save_xarray(xarray, frames_per_split):
        total_frames = xarray.sizes['frame']
        num_splits = (total_frames + frames_per_split - 1) // frames_per_split  # Calculate number of splits

        for i in range(num_splits):
            start_frame = i * frames_per_split
            end_frame = min(start_frame + frames_per_split, total_frames)
            
            
            split_xarray = xarray.sel(frame=slice(start_frame, end_frame))
            
            # Construct a file name for each split
            split_file_name = f"part{i+1}.nc"
            
            # Save the split xarray
            #Y_split = save_minian(Y.rename("split_xarray"), dpath=intpath, overwrite=True)
            future = client.submit(process_chunk, split_xarray, i, chk, param_save_minian, param_dict)
            chunk_futures.append(future)

        return chunk_futures

    split_and_save_xarray(Y, frames_per_split)
    results = client.gather(chunk_futures)
    client.close()
    cluster.close()
   

    elapsed_time = time.time() - start_time_total     # End the timer and print the elapsed time
    print(f"Total Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 16 Complete\033[0m")

    print('Analysis Complete')