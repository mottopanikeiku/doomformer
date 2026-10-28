"""Rebuild digits8x8.npz from scikit-learn's copy of the UCI "Optical Recognition of Handwritten Digits" test set.

    pip install scikit-learn && python models/digits/make_digits.py          # scikit-learn is not a doomformer dependency

images: 1,797 x 8 x 8 uint8, grey levels 0..16 (counts of set pixels in 4x4 blocks of the original 32x32 bitmaps).
labels: 1,797 uint8 digits 0..9.
heldout: 500 digits (50 per class, chosen with numpy.random.default_rng(0) per class, in class order) the learner never
trains on; the other 1,297 are the training pool. The split is stored, so it does not depend on the NumPy version.
"""
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

d = load_digits()
images, labels = d.images.astype(np.uint8), d.target.astype(np.uint8)
assert images.shape == (1797, 8, 8) and images.max() == 16
rng = np.random.default_rng(0)
heldout = np.zeros(len(labels), dtype=bool)
for c in range(10):
    idx = rng.permutation(np.where(labels == c)[0])
    heldout[idx[:50]] = True
np.savez_compressed(Path(__file__).with_name("digits8x8.npz"), images=images, labels=labels, heldout=heldout)
print(images.shape, int(heldout.sum()), "held out")
