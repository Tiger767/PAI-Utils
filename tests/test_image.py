"""
Author: Travis Hammond
Version: 12_9_2020
"""


from paiutils.image import *


def test_rgb2bgr():
    x = np.random.randint(0, 255, size=(32, 32, 3), dtype='uint8')
    assert (bgr2rgb(rgb2bgr(x)) == x).all()

def test_bgr2rgb():
    x = np.random.randint(0, 255, size=(32, 32, 3), dtype='uint8')
    assert (rgb2bgr(bgr2rgb(x)) == x).all()

def test_bgr2hsv():
    x = np.random.randint(0, 255, size=(32, 32, 3), dtype='uint8')
    assert np.abs(
        hsv2bgr(bgr2hsv(x)).astype('int') - x.astype('int')
    ).mean() < 5

def test_hsv2bgr():
    x = np.random.randint(0, 180, size=(32, 32, 3), dtype='uint8')
    assert np.abs(
        bgr2hsv(hsv2bgr(x)).astype('int') - x.astype('int')
    ).mean() < 5

def test_bgr2hls():
    x = np.random.randint(0, 180, size=(32, 32, 3), dtype='uint8')
    assert np.abs(
        hls2bgr(bgr2hls(x)).astype('int') - x.astype('int')
    ).mean() < 5

def test_hls2bgr():
    x = np.random.randint(0, 180, size=(32, 32, 3), dtype='uint8')
    assert np.abs(
        bgr2hls(hls2bgr(x)).astype('int') - x.astype('int')
    ).mean() < 5

def test_gray():
    x = np.random.randint(0, 255, size=(32, 32, 3), dtype='uint8')
    assert len(gray(x).shape) == 2

def test_resize():
    x = np.random.randint(0, 255, size=(32, 32, 3), dtype='uint8')
    x = resize(x, (256, 256))
    assert x.shape == (256, 256, 3)
    x = resize(x, (32, 32))
    assert x.shape == (32, 32, 3)

def test_normalize():
    x = np.random.randint(0, 255, size=(32, 32, 3), dtype='uint8')
    x = normalize(x)
    assert (-1 <= x).all()
    assert (x <= 1).all()

def test_denormalize():
    x = np.random.randint(0, 255, size=(32, 32, 3), dtype='uint8')
    x2 = denormalize(normalize(x))
    assert (0 <= x2).all()
    assert (x2 <= 255).all()
    assert np.sum(np.abs(x - x2)) < .0000001 

def test_pyr():
    x = np.random.randint(0, 255, size=(32, 32, 3), dtype='uint8')
    x2 = pyr(x, 2)
    assert x2.shape == (128, 128, 3)
    x2 = pyr(x2, -2)
    assert x2.shape == (32, 32, 3)

def test_load():
    pass

def test_save():
    pass

def test_increase_brightness():
    pass

def test_set_brightness():
    pass

def test_set_gamma():
    pass

def test_apply_clahe():
    pass

def test_equalize():
    pass

def test_rotate():
    pass

def test_hflip():
    pass

def test_vflip():
    pass

def test_translate():
    pass

def test_crop_rect():
    pass

def test_crop_rect_coords():
    pass

def test_shrink_sides():
    pass

def test_crop():
    pass

def test_pad():
    pass

def test_blend():
    pass

def test_zoom():
    pass

def test_transform_perspective():
    pass

def test_unsharp_mask():
    pass

def test_create_mask_of_colors_in_range():
    pass

def test_compute_color_ranges():
    pass

def test_create_magnitude_spectrum():
    pass

def test_freq_filter_image():
    pass

def test_create_histograms():
    pass

def test_histogram_back_projector():
    pass

def test_template_matecher():
    pass

def test_camera():
    pass
    
def test_lock_dict():
    pass

def test_windows():
    pass
