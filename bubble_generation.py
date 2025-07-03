"""
bubble_generation.py

This script generates a synthetic 3D binary ionization field (bubble field) to test
mean free path (MFP) methods. It produces a simplified scenario
of ionized 'bubbles' in a neutral medium, without using a corresponding density field.

Intended for testing and comparing mfp_old, mfp_new, and mfp_seq methods.

Author: R.R.Crawford
Date: 03/07.2025

"""
import numpy as np
from numba import njit, prange
import time

# Parameters
n = 120  # Grid size
boxsize = 200  # Mpc
r_mpc = 10  # Physical radius
num_bubbles = 5000  # Number of bubbles

vox_size = boxsize / n
r_vox = r_mpc / vox_size
radius = int(np.ceil(r_vox))

arr3d1 = np.ones((n, n, n), dtype=np.float32)

np.random.seed(42)
centers = np.random.randint(radius, n, size=(num_bubbles, 3))

@njit(parallel=True)
def random_bubbles(arr, centers, r_vox):
    n = arr.shape[0]
    for i in prange(centers.shape[0]):
        cx, cy, cz = centers[i]
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                for dz in range(-radius, radius+1):
                    r2 = dx*dx + dy*dy + dz*dz
                    if r2 <= r_vox * r_vox:
                        r = np.sqrt(r2)
                        prof = 1 / (1.0 + (r / r_vox)**8)
                        x = (cx + dx) % n
                        y = (cy + dy) % n
                        z = (cz + dz) % n
                        arr[x, y, z] = min(arr[x, y, z], 1.0 - prof)


start = time.time()
random_bubbles(arr3d1, centers, r_vox)
end = time.time()

xHII = 1 - np.mean(arr3d1)

print(f"Done. Time taken: {end - start:.2f} seconds")
print(f"Final xHII: {xHII:0.3f}")

filename = f'bubbles_r{r_mpc}Mpc_n8_xHII{xHII:0.3f}.npy'
np.save(filename, arr3d1)
print(f'Saved to {filename}')
