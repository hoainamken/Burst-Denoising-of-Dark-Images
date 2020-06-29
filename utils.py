import numpy as np

#ref: https://github.com/cchen156/Learning-to-See-in-the-Dark
def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def pack_raw_quad(raw):
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 0, 0) / (16383 - 0)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    out = np.concatenate((im[1:H:2, 0:W:2, :], 
                          im[1:H:2, 1:W:2, :], 
                          im[0:H:2, 1:W:2, :], 
                          im[0:H:2, 0:W:2, :]), axis=2)
    return out