# models/digits: 8x8 handwritten digits for `doomformer learn digits`

`digits8x8.npz` holds 1,797 digits:

* `images`: 1,797 x 8 x 8 uint8, grey levels 0..16.
* `labels`: 0..9.
* `heldout`: 500 digits, 50 per class, that the learner never trains on.

Source: the test set of the UCI "Optical Recognition of Handwritten Digits" data set (E. Alpaydin, C. Kaynak, 1998,
https://doi.org/10.24432/C50P49), via scikit-learn 1.9.1 `load_digits`. Licence: CC BY 4.0. Its 13 writers are not
among the 30 of the training set. `make_digits.py` rebuilds the file. The split is stored in the file.

`doomformer learn digits` uses 37 binary features: the centre 6x6 pixels thresholded at grey level 8, plus a bias.

`numpy_seeds.py` runs the same learner in NumPy on seeds 0 to 19. Seed 0 is the stream of the DOOM run in
`learn_1500.log`.
