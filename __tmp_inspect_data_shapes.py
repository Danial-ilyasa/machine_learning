import numpy as np
import os
path = 'c:/main/jst'
for fn in ['ti.npy','tl.npy','test_images.npy','test_labels.npy']:
    a = np.load(os.path.join(path, fn))
    print(fn, a.shape, a.dtype)
