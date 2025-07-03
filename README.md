# Modified MFP Bubble Size Distribution Methods

This repository contains code to compare different mean free path (MFP) estimation techniques for identifying ionized bubble size distributions (BSDs) in cosmological simulations.  

It includes:
- `mfp_old.py`: Implementation based on [Mesinger & Furlanetto (2007)](https://ui.adsabs.harvard.edu/abs/2007ApJ...669..663M/abstract)
- `mfp_new.py`: A new method with fixed-direction 3D sampling
- `mfp_seq.py`: A sequential ray method based on `mfp_new.py`

## Getting Started

To run the examples:

1. Clone the repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run `examples/test_run.ipynb`

## License & Citation

See `CITATIONS.md` and `LICENSE`.
