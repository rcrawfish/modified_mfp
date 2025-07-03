import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
from datetime import datetime
from time import time, sleep
from tqdm import tqdm

def mfp3d_new(arr, xth=0.5, iterations=10000000, verbose=True, point='random'):
    iterations = int(iterations)
    
    if verbose: print('Initializing random rays...', end=' ')
    
    info = arr.shape
    longest = max(info)
    num_sz  = np.zeros(longest)
    ar = np.zeros(info, dtype=np.float64)
    ar[arr >= xth] = 1
    
    # Define orthogonal unit vectors in each direction
    unit_vectors = np.array([
        [1, 0, 0], [-1, 0, 0],  # +x, -x
        [0, 1, 0], [0, -1, 0],  # +y, -y
        [0, 0, 1], [0, 0, -1]   # +z, -z
    ])

    if point == 'random':
        loc = np.argwhere(ar == 1)
        rand_loc = np.random.randint(0, high=loc.shape[0], size=iterations)
        xs, ys, zs = loc[rand_loc, 0], loc[rand_loc, 1], loc[rand_loc, 2]
    else:
        xs, ys, zs = point
        if ar[xs, ys, zs] == 0:
            print('Given point is outside the structure.')
            return None
        xs, ys, zs = np.full(iterations, xs), np.full(iterations, ys), np.full(iterations, zs)

    if verbose: print('done')
    
    sizes = []
    
    with tqdm(total=iterations, dynamic_ncols=False, disable=not verbose) as pbar:
        for i in range(iterations):
            x0, y0, z0 = xs[i], ys[i], zs[i]  # Get the random starting point

            # Shoot rays in each of the six directions
            for dx, dy, dz in unit_vectors:
                x, y, z = x0, y0, z0  # Reset position for each ray direction
                step = 0
                
                # Make sure each ray is still within the grid and in an HII region
                while 0 <= x < info[0] and 0 <= y < info[1] and 0 <= z < info[2] and ar[int(x), int(y), int(z)] > 0.5:
                    x += dx
                    y += dy
                    z += dz
                    step += 1
                
                radius.append(step)

            # Compute the radii from the full lengths
            rx = (radius[0] + radius[1]) / 2
            ry = (radius[2] + radius[3]) / 2
            rz = (radius[4] + radius[5]) / 2
            
            # Compute the radius vector from r = √(x^2 + y^2 + z^2)
            rr = np.sqrt(rx**2 + ry**2 + rz**2)
            
            sizes.append(rr)

            # Update progress bar
            pbar.update(1)
    
    # Convert to NumPy array and bin the array
    sizes = np.array(sizes)
    hist, bins = np.histogram(sizes, bins=np.arange(longest))
    
    return hist, bins[:-1]

def mfp(data, xth=0.5, boxsize=None, iterations = 10000000, verbose=True, upper_lim=False, bins=None, r_min=None, r_max=None):
    """
    Computes the mean free path (MFP) distribution for a 3D binary field.
    

    Parameters
    ----------
    data : ndarray
        A 3D numpy array representing the ionization (or similar) field. 
        Values above `xth` define the ROI (e.g., ionized regions).
    xth : float, optional
        Threshold value to define the ROI. Default is 0.5.
    boxsize : float, optional
        Physical size of the box in Mpc. If not provided, a default from `conv.LB` is used.
    iterations : int, optional
        Number of rays or samples to draw for the MFP computation. Default is 10 million.
    verbose : bool, optional
        If True, prints progress and runtime info. Default is True.
    upper_lim : bool, optional
        If True, inverts the threshold to compute the MFP through low-value regions (e.g., neutral instead of ionized).
    bins : int, optional
        If provided, rebins the final distribution into this number of bins.
    r_min : float, optional
        Minimum radius for rebinning. Only used if `bins` is set.
    r_max : float, optional
        Maximum radius for rebinning. Only used if `bins` is set.

    Returns
    -------
    r0 : ndarray
        Array of physical distances (in Mpc) corresponding to bin centers.
    p0 : ndarray
        Normalized probability distribution function (PDF) of the mean free path distances.
    """
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
    if dim == 3:
        if verbose: print("*New* MFP method applied on 3D data.")
        out = mfp3d_new(data, xth, iterations=iterations, verbose=verbose)
    else:
        if verbose: print("The data doesn't have the correct dimension")
        return 0
    
    nn = out[0]/(2*iterations)
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
    
    r0,p0 = rr*boxsize/data.shape[0]/np.sqrt(3), rr*nn
    
    # Normalize pdf to 1
    norm = np.trapz(p0, r0)
    p0 /= norm

    if bins is not None: r0,p0 = rebin_bsd(r0, p0, bins=bins, r_min=r_min, r_max=r_max)
    return r0, p0
