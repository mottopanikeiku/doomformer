# models/digits: 8x8 handwritten digits for `doomformer learn digits`

`digits8x8.npz` holds 1,797 digits (`images` 1,797 x 8 x 8 uint8, grey levels 0..16; `labels` 0..9; `heldout`, 500 digits,
50 per class, that the learner never trains on).

Source: the test set of the UCI "Optical Recognition of Handwritten Digits" data set (E. Alpaydin, C. Kaynak, 1998,
https://doi.org/10.24432/C50P49), as copied into scikit-learn (`sklearn.datasets.load_digits`, scikit-learn 1.9.1). The
UCI page lists the licence as Creative Commons Attribution 4.0 (CC BY 4.0); credit: Alpaydin and Kaynak. The 13 writers of
the test set are different people from the 30 of the original training set. 4x4-block counts of NIST 32x32 bitmaps give the
8x8 grey levels. `make_digits.py` rebuilds the file (scikit-learn is not a doomformer dependency). The split is stored in
the file, so it does not depend on the NumPy version.

`doomformer learn digits` crops the 8x8 image to its centre 6x6 pixels, sets a feature to 1 where the grey level is at
least 8, and adds a constant 1 feature (a bias): 37 binary features.

`numpy_seeds.py` (NumPy only, no DOOM) runs the same learner on 20 example streams, seeds 0 to 19. Seed 0 is the stream of the
DOOM run in `learn_1500.log`, so it is one of the 20 (the other 19 are not DOOM runs).
