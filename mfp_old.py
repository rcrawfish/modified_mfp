"""
This file implements the MFP (mean free path) method from:

    Mesinger, A., & Furlanetto, S. R. (2007),
    "Efficient Simulations of Early Structure Formation and Reionization",
    The Astrophysical Journal, 669(2), 663–675.
    https://doi.org/10.1086/521806

Original code from the tools21cm package:
    https://github.com/21cmfast/tools21cm

Modifications and usage here are for research and educational purposes only.
All credit for the original method and implementation belongs to the original authors.

If you use this code, please cite the above paper and consider referencing tools21cm.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
from datetime import datetime
from time import time, sleep
from tqdm import tqdm

def mfp3d(arr, xth=0.5, iterations=10000000, verbose=True, point='random'):
    iterations = int(iterations)

    if verbose: print('Initialising random rays...', end=' ')
    
    info = arr.shape
    longest = max(info)
    num_sz  = np.zeros(longest)
    ar = np.zeros(info, dtype=np.float64)
    ar[arr >= xth] = 1

    thetas = np.random.randint(0, 360, size=iterations)
    phis   = np.random.randint(0, 360, size=iterations)
    
    # Precompute trigonometric values
    sin_thetas = np.sin(np.radians(thetas))
    cos_thetas = np.cos(np.radians(thetas))
    cos_phis   = np.cos(np.radians(phis))
    sin_phis   = np.sin(np.radians(phis))
    
    ls = sin_thetas * cos_phis
    ms = sin_thetas * sin_phis
    ns = cos_thetas

    if point == 'random':
        loc = np.argwhere(ar == 1)
        rand_loc = np.random.randint(0, high=loc.shape[0], size=iterations)
        xs, ys, zs = loc[rand_loc, 0], loc[rand_loc, 1], loc[rand_loc, 2]
    else:
        xs, ys, zs = point
        if ar[xs, ys, zs] == 0:
            print('Given point is outside the structure.')
            return None
        xs, ys, zs = np.full(iterations, xs, dtype=np.float64), np.full(iterations, ys, dtype=np.float64), np.full(iterations, zs, dtype=np.float64)
    
    # Ensure coordinates are of type float64 for compatibility with interpolation
    xs, ys, zs = xs.astype(np.float64), ys.astype(np.float64), zs.astype(np.float64)

    interp_func = RegularGridInterpolator(
        (np.arange(info[0]), np.arange(info[1]), np.arange(info[2])),
        ar,
        bounds_error=False,
        fill_value=0
    )

    if verbose:
        print('done')
        print('Estimating ray lengths...')
    
    sleep(0.01)
    total_iterations = longest
    with tqdm(total=total_iterations, dynamic_ncols=False, disable=not verbose) as pbar:
        for rr in range(longest):        # One ray for each point from 0 --> longest axis of 3D array
            
            # Move points one step along each axis
            xs += ls
            ys += ms
            zs += ns
            
            # Efficiently create points and interpolate values
            pts = np.column_stack((xs, ys, zs))
            vals = interp_func(pts)
            
            # Use boolean indexing instead of np.argwhere and np.delete
            valid = vals > 0.5                            # True if still in ROI
            num_sz[rr] = len(xs) - np.sum(valid)          # Number of rays exiting ROI at length rr
            xs, ys, zs = xs[valid], ys[valid], zs[valid]  # Only keep points still in ROI
            ls, ms, ns = ls[valid], ms[valid], ns[valid]  # Only keep rays still in ROI
            
            pbar.update(1)  # Increment the progress bar
            if len(xs) == 0:
                pbar.n = pbar.total  # Manually set the progress to 100%
                pbar.refresh()  # Refresh the bar to show the update
                break
        # pbar.set_postfix({'Completion': '100%'})

    size_px = np.arange(longest)   # Bins for distribution (one for each possible ray length)
    return num_sz, size_px

def mfp2d(arr, xth=0.5, iterations=1000000, verbose=True, point='random'):
    iterations = int(iterations)
    
    if verbose: print('Initializing random rays...', end=' ')
    
    info = arr.shape
    longest = int(np.sqrt(2) * max(info))
    num_sz  = np.zeros(longest)

    ar = np.zeros(info, dtype=np.float64)
    ar[arr >= xth] = 1

    thetas = np.random.randint(0, 360, size=iterations)
    
    # Precompute trigonometric values
    sin_thetas = np.sin(np.radians(thetas))
    cos_thetas = np.cos(np.radians(thetas))

    ls = sin_thetas
    ms = cos_thetas

    if point == 'random':
        loc = np.argwhere(ar == 1)
        rand_loc = np.random.randint(0, high=loc.shape[0], size=iterations)
        xs, ys = loc[rand_loc, 0], loc[rand_loc, 1]
    else:
        xs, ys = point
        if ar[xs, ys] == 0:
            print('Given point is outside the structure.')
            return None
        xs, ys = np.full(iterations, xs, dtype=np.float64), np.full(iterations, ys, dtype=np.float64)
    
    # Ensure coordinates are of type float64 for compatibility with interpolation
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)

    interp_func = RegularGridInterpolator(
        (np.arange(info[0]), np.arange(info[1])),
        ar,
        bounds_error=False,
        fill_value=0
    )

    if verbose:
        print('done')
        print('Estimating ray lengths...')
    
    sleep(0.01)
    total_iterations = longest
    with tqdm(total=total_iterations, dynamic_ncols=False, disable=not verbose) as pbar:
        for rr in range(longest):
            xs += ls
            ys += ms
            
            # Efficiently create points and interpolate values
            pts = np.column_stack((xs, ys))
            vals = interp_func(pts)
            
            # Use boolean indexing instead of np.argwhere and np.delete
            valid = vals > 0.5
            num_sz[rr] = len(xs) - np.sum(valid)
            xs, ys = xs[valid], ys[valid]
            ls, ms = ls[valid], ms[valid]
            
            pbar.update(1)  # Increment the progress bar
            if len(xs) == 0:
                pbar.n = pbar.total  # Manually set the progress to 100%
                pbar.refresh()  # Refresh the bar to show the update
                break
        # pbar.set_postfix({'Completion': '100%'})

    size_px = np.arange(longest)
    return num_sz, size_px

def mfp(data, xth=0.5, boxsize=None, iterations = 10000000, verbose=True, upper_lim=False, bins=None, r_min=None, r_max=None):
    if boxsize is None:
        boxsize = conv.LB
        print('Boxsize is set to %.2f Mpc.'%boxsize) 
    dim = len(data.shape)
    t1 = datetime.now()
    if (upper_lim): 
        data = -1.*data
        xth  = -1.*xth
    check_box = (data>=xth).sum()
    if verbose:
        print(f'{check_box}/{data.size} cells are marked as region of interest (ROI).')
    if check_box==0:
        data = np.ones(data.shape)
        iterations = 3
    if dim == 2:
        if verbose: print("*Old* MFP method applied on 2D data.")
        out = mfp2d(data, xth, iterations=iterations, verbose=verbose)
    elif dim == 3:
        if verbose: print("*Old* MFP method applied on 3D data.")
        out = mfp3d(data, xth, iterations=iterations, verbose=verbose)
    else:
        if verbose: print("The data doesn't have the correct dimension")
        return 0
    nn = out[0]/iterations
    rr = out[1]
    t2 = datetime.now()
    runtime = (t2-t1).total_seconds()/60

    if verbose: print("\nProgram runtime: %.2f minutes." %runtime)
    if check_box==0:
        if verbose: print("There is no ROI in the data. Therefore, the number density of all the sizes are zero.")
        # return rr*boxsize/data.shape[0], np.zeros(rr.shape)
        nn = np.zeros(rr.shape)
    if verbose: print("The output contains a tuple with three values: r, rdP/dr")
    if verbose: print("The curve has been normalized.")

    r0,p0 = rr*boxsize/data.shape[0], rr*nn #rr[nn.argmax()]*boxsize/data.shape[0]
    
    norm = np.trapz(p0,r0)
    p0 /= norm
    
    if bins is not None: r0,p0 = rebin_bsd(r0, p0, bins=bins, r_min=r_min, r_max=r_max)
    return r0, p0
