"""
Author: Travis Hammond
Version: 11_24_2020
"""


import cv2
import numpy as np
from time import time, sleep
from threading import Thread, Event, Lock
from matplotlib import pyplot as plt


def rgb2bgr(image):
    """Converts a RGB image to a BGR image.
    params:
        image: A numpy ndarray, which has 3 dimensions
    return: A numpy ndarray, which has 3 dimensions
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr2rgb(image):
    """Converts a BGR image to a RGB image.
    params:
        image: A numpy ndarray, which has 3 dimensions
    return: A numpy ndarray, which has 3 dimensions
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr2hsv(image):
    """Converts a BGR image to a HSV image.
    params:
        image: A numpy ndarray, which has 3 dimensions
    return: A numpy ndarray, which has 3 dimensions
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hsv2bgr(image):
    """Converts a HSV image to a BGR image.
    params:
        image: A numpy ndarray, which has 3 dimensions
    return: A numpy ndarray, which has 3 dimensions
    """
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def bgr2hsl(image):
    """Converts a BGR image to a HSL image.
    params:
        image: A numpy ndarray, which has 3 dimensions
    return: A numpy ndarray, which has 3 dimensions
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSL)


def hls2bgr(image):
    """Converts a HLS image to a BGR image.
    params:
        image: A numpy ndarray, which has 3 dimensions
    return: A numpy ndarray, which has 3 dimensions
    """
    return cv2.cvtColor(image, cv2.COLOR_HLS2BGR)


def gray(image):
    """Converts a BGR image to a grayscale image.
    params:
        image: A numpy ndarray, which has 3 dimensions
    return: A numpy ndarray, which has 2 dimensions
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize(image, target_shape, interpolation=None):
    """Resizes an image to a targeted shape.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        target_shape: A tuple with the vertical size then horizontal size
        interpolation: A cv2 interpolation
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    if interpolation:
        return cv2.resize(image, target_shape, interpolation=interpolation)
    if np.prod(image.shape) > np.prod(target_shape):
        return cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA)
    return cv2.resize(image, target_shape, interpolation=cv2.INTER_CUBIC)


def normalize(image):
    """Normalizes an image between -1 and 1.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    return (image.astype(np.float) - 127.5) / 127.5


def denormalize(image):
    """Denormalizes an image that is between -1 and 1 to 0 and 255.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions and is normalized
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    return np.clip(image * 127.5 + 127.5, 0, 255).astype(np.uint8)


def pyr(image, level):
    """Resize image using pyramids.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        level: An integer, which if positive enlarges and if negative reduces
    returns: A numpy ndarray, which has the same number of dimensions
             of the image
    """
    if level > 0:
        for _ in range(level):
            image = cv2.pyrUp(image)
    elif level < 0:
        for _ in range(-level):
            image = cv2.pyrDown(image)
    return image


def load(filename, target_shape=None, color=True):
    """Loads an image from a file.
    params:
        filename: A string, which is the directory or filename of the
                  file to load
        target_shape: A tuple with the vertical size then horizontal size
        color: A boolean, which determines if the image should be
               converted to gray scale
    return: A numpy ndarray, which has 2 or 3 dimensions
    """
    image = cv2.imread(filename)
    if image is None:
        raise ValueError(f'{filename} is not a supported image file')
    if target_shape is not None:
        image = resize(image, target_shape)
    if not color:
        image = gray(image)
    return image


def save(filename, image, target_shape=None, color=True):
    """Saves an image to a file.
    params:
        filename: A string, which is the directory or filename to save image to
        image: A numpy ndarray, which has 2 or 3 dimensions
        target_shape: A tuple with the vertical size then horizontal size
        color: A boolean, which determines if the image should be
               converted to gray scale
    """
    if target_shape is not None:
        image = resize(image, target_shape)
    if not color:
        image = gray(image)
    cv2.imwrite(filename, image)


def increase_brightness(image, percentage, relative=False):
    """Increases the brightness of image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        percentage: An integer, which is how much to increase
        relative: A boolean, which determines if the percentage is
                  is in terms of max brightness or current brightness
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    if relative:
        v = v.astype(np.int) + v.astype(np.int) * percentage / 100
    else:
        v = v.astype(np.int) + 255 * percentage / 100
    v = v.round().clip(0, 255).astype(np.uint8)
    hsv[:, :, 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def set_brightness(image, percentage, relative=False):
    """Sets the brightness of image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        percentage: An integer, which is how much to increase
        relative: A boolean, which determines if the percentage is
                  is in terms of max brightness or current brightness
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    if relative:
        v = v.astype(np.int) * percentage / 100
        v = v.round().clip(0, 255)
    else:
        v = np.full(v.shape, np.clip(round(255 * percentage / 100), 0, 255))
    v = v.astype(np.uint8)
    hsv[:, :, 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def set_gamma(image, gamma=1.0):
    """Set gamma levels of the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        gamma: A float, which is the amount to change the images gamma
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    image = (image / 255.0)**(1.0 / gamma) * 255
    return image.round().astype(np.uint8)


def apply_clahe(image, clip_limit=40.0, tile_grid_size=(8, 8)):
    """Applys CLAHE (Contrast Limited Adaptive Histogram Equalization).
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions (preferably 2)
        clip_limit: A float, which is the threshold for contrasting
        tile_grid_size: A tuple of 2 natural numbers, which is the number
                        of rows and columns, respectively
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    else:
        b = clahe.apply(image[:, :, 0])
        g = clahe.apply(image[:, :, 1])
        r = clahe.apply(image[:, :, 2])
        return cv2.merge((b, g, r))


def equalize(image):
    """Equalizes the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    else:
        b = cv2.equalizeHist(image[:, :, 0])
        g = cv2.equalizeHist(image[:, :, 1])
        r = cv2.equalizeHist(image[:, :, 2])
        return cv2.merge((b, g, r))


def rotate(image, angle):
    """Rotates the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        angle: A float, which is in terms of degress
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
    )


def hflip(image):
    """Horizontally flips the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    return cv2.flip(image, 1)


def vflip(image):
    """Vertically flips the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    return cv2.flip(image, 0)


def translate(image, vertical=0, horizontal=0):
    """Translates the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        vertical: An integer (possibly a float), which is the amount to
                  shift the image vertically
        horizontal: An integer (possibly a float), which is the amount to
                  shift the image horizontally
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    return cv2.warpAffine(image,
                          np.float32([[1, 0, horizontal], [0, 1, vertical]]),
                          image.shape[1::-1])


def crop_rect(image, vertical, horizontal, width, height):
    """Crops a rectangle out of the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        vertical: An integer, which is the vertical coord for the top left
                  of the rectangle
        horizontal: An integer, which is the horizontal coord for the top left
                  of the rectangle
        width: An integer, which is the width of the rectangle
        height: An integer, which is the height of the rectangle
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    return image[vertical:vertical + height, horizontal:horizontal + width]


def crop_rect_coords(image, vertical1, horizontal1, vertical2, horizontal2):
    """Crops a rectangle out of the image through two coords.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        vertical1: An integer, which is the vertical coord for the top left
                   of the rectangle
        horizontal1: An integer, which is the horizontal coord for the top left
                     of the rectangle
        vertical2: An integer, which is the vertical coord for the bottom right
                   of the rectangle
        horizontal2: An integer, which is the horizontal coord for the bottom
                     right of the rectangle
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    return image[vertical1:vertical2, horizontal1:horizontal2]


def shrink_sides(image, ts=0, bs=0, ls=0, rs=0):
    """Shrinks/crops the image through shrinking each side of the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        ts: An integer, which is the amount to shrink the top side
            of the image
        bs: An integer, which is the amount to shrink the bottom side
            of the image
        ls: An integer, which is the amount to shrink the left side
            of the image
        rs: An integer, which is the amount to shrink the right side
            of the image
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    return image[ts:image.shape[0] - bs, ls:image.shape[1] - rs]


def crop(image, shape, horizontal_center=0, vertical_center=0):
    """Crops the image with a given center coord and the shape of a rectangle.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        shape: A tuple of 2 integers, which is the shape of the rectangle
        horizontal_center: An integer, which is the offset from the
                           image's horizontal center
        vertical_center: An integer, which is the  offset from the
                         image's vertical center
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    ds = (image.shape[0] - shape[0]) // 2, (image.shape[1] - shape[1]) // 2
    return shrink_sides(image, ds[0] + vertical_center,
                        ds[0] - vertical_center,
                        ds[1] + horizontal_center,
                        ds[1] - horizontal_center)


def pad(image, ts=0, bs=0, ls=0, rs=0, color=(0, 0, 0)):
    """Pads the image through adding pixels to each side of the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        ts: An integer, which is the amount to pad the top side
            of the image
        bs: An integer, which is the amount to pad the bottom side
            of the image
        ls: An integer, which is the amount to pad the left side
            of the image
        rs: An integer, which is the amount to pad the right side
            of the image
        color: A tuple of 3 integers or an integer with a range of
               0-255 (inclusive), which is the color of the padding
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    return cv2.copyMakeBorder(image, ts, bs, ls, rs, cv2.BORDER_CONSTANT,
                              value=color)


def blend(image1, image2, image1_weight=.5, image2_weight=None):
    """Blends two images together.
    params:
        image1: A numpy ndarray, which has 2 or 3 dimensions
        image2: A numpy ndarray, which has same dimensions as image1
        image1_weight: A float, which is the intensity of image1
                       to preserve
        image2_weight: A float, which is the intensity of image2
                       to preserve
    return: A numpy ndarray, which has the same number of dimensions as image1
    """
    if image2_weight is None:
        image2_weight = 1 - image1_weight
    return cv2.addWeighted(image1, image1_weight, image2, image2_weight, 0)


def zoom(image, shape, horizontal_center=0, vertical_center=0):
    """Zooms the image to shape on a given center coord
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        shape: A tuple of 2 integers, which is the shape of the zoomed image
        horizontal_center: An integer, which is the horizontal offset from
                           the image's center
        vertical_center: An integer, which is the vertical offset from
                         the image's center
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    old_shape = image.shape[1::-1]
    if old_shape[0] < shape[0] and old_shape[1] < shape[1]:
        ds = (shape[0] - old_shape[0]) // 2, (shape[1] - old_shape[1]) // 2
        image = pad(image, ds[0] + vertical_center,
                    ds[0] - vertical_center,
                    ds[1] + horizontal_center,
                    ds[1] - horizontal_center)
        return resize(image, old_shape)
    else:
        image = crop(image, shape, vertical_center=vertical_center,
                    horizontal_center=horizontal_center)
        return resize(image, old_shape)


def transform_perspective(image, pts, shape):
    """Transforms the perspective of an image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        pts: A list of list with 2 integers (possibly floats)
        shape: A tuple of 2 integers
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    pts1 = np.float32(pts)
    pts2 = np.float32([[0, 0], [shape[0], 0], [0, shape[1]], shape])
    m = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, m, shape)


def unsharp_mask(image, kernel_shape=(5, 5), sigma=1.0,
                 amount=1.0, threshold=0):
    """Sharpens the image through the unsharp masking technique.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions
        kernel_shape: A tuple of 2 integers, which is the shape of the
                      blurring kernel
        sigma: A float, which is the standard deviation of the Gaussian blur
        amount: A float, which is the amount to subtracted the blurred
                image from the image
        threshold: An integer within 0-255 (inclusive), which is the low
                   contrast threshold to copy the image to the sharpened image
    return: A numpy ndarray, which has the same number of dimensions as image
    """
    blurred = cv2.GaussianBlur(image, kernel_shape, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def create_mask_of_colors_in_range(image, lower_bounds, upper_bounds):
    """Creates a mask of the colors in within the lower and upper bounds.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions (BGR)
        lower_bounds: A tuple of 3 integers (HSV), which is the lower bound
        upper_bounds: A tuple of 3 integers (HSV), which is the upper bound
    return: A numpy ndarray, which has 2 dimensions
    """
    if isinstance(lower_bounds[0], int):
        lower_bounds = [lower_bounds]
    if isinstance(upper_bounds[0], int):
        upper_bounds = [upper_bounds]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
        mask = cv2.inRange(hsv, tuple(lower_bound), tuple(upper_bound))
        masks = cv2.bitwise_or(masks, mask)
    return masks


def compute_color_ranges(images, percentage_captured=50,
                         num_bounds=1, use_evolution_algo=False):
    """Computes the color ranges that captures a percentage of the image.
    This algorithm is not well designed and is mainly for testing purposes.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions (BGR)
        percentage_captured: An integer within 0-100 (inclusive), which acts
                             as the threshold for the bounds
        num_bounds: An integer, which does nothing
        use_evolution_algo: A boolean, which does nothing
    return: A tuple of 2 list with the former containing lower
            bounds and the latter upper bounds
    """
    if use_evolution_algo:
        raise NotImplementedError('NOT DONE')
    else:
        hsv = cv2.cvtColor(np.vstack(images), cv2.COLOR_BGR2HSV)
        hsv_flat = np.reshape(hsv, (np.prod(hsv.shape[:2]), 3))
        pc = percentage_captured / 2
        lower_bounds = [np.floor(np.percentile(hsv_flat, 50 - pc, axis=0))]
        upper_bounds = [np.ceil(np.percentile(hsv_flat, 50 + pc, axis=0))]
        # should do, but slow: greatest number, shortest range
    return lower_bounds, upper_bounds


def create_magnitude_spectrum(image):
    """Creates a magnitude spectrum from image
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions (BGR)
    return: A numpy ndarray, which has 2 dimensions
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0],
                                                   dft_shift[:, :, 1]))
    return magnitude_spectrum


def freq_filter_image(image, high=True):
    """Filters frequencies in the image.
    params:
        image: A numpy ndarray, which has 2 or 3 dimensions (BGR)
    return: A numpy ndarray, which has 2 dimensions
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    if high:
        dft_shift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0
    else:
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
        dft_shift *= mask
    image = cv2.idft(np.fft.ifftshift(dft_shift))
    image = cv2.magnitude(image[:, :, 0], image[:, :, 1])
    return image


def create_histograms(images, hsv_images=False, channels=None,
                      vrange=None, num_bins=None):
    """
    params:
        image: A list of numpy ndarray, which each ndarray is 2 or
               3 dimensions (must all have same dimensions)
        hsv_images: A boolean, which determines if the image is HSV
        channels: A list of integers within 0-2 (inclusive), which are the
                  channels to get the histograms of
        vrange: A list the same length as channels with list containing 2
                integers containing the lower and upper+1 value of a channel
        num_bins: An integer, which is the number of bins to have for
                  the histograms
    return: A numpy ndarray, which is a list of the histograms
    """
    if not isinstance(images, list):
        images = [images]
    if channels is None:
        if images[0].ndim == 2:
            channels = [0]
        else:
            channels = list(range(images[0].shape[-1]))
    if vrange is None:
        if hsv_images:
            vrange = []
            if 0 in channels:
                vrange += [0, 180]
            if 1 in channels:
                vrange += [0, 256]
            if 2 in channels:
                vrange += [0, 256]
        else:
            vrange = [0, 256] * len(channels)
    if num_bins is None:
        if hsv_images:
            num_bins = []
            if 0 in channels:
                num_bins += [180]
            if 1 in channels:
                num_bins += [256]
            if 2 in channels:
                num_bins += [256]
        else:
            num_bins = [256] * len(channels)
    else:
        num_bins = [num_bins]
    return cv2.calcHist(images, channels, None, num_bins, vrange)


class HistogramBackProjector:
    """This Class is used to find objects of interest in an image"""

    def __init__(self, object_image):
        """Initializes the HBP by computing the object image's histogram.
        params:
            object_image: A numpy ndarray, which has 3 dimensions (BGR)
        """
        hsv = cv2.cvtColor(object_image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self.hist = hist

    def backproject(self, image, raw=False, threshold=50, disc_kernel=(5, 5)):
        """Back projects the image to the object image
        params:
            image: A numpy ndarray, which has 3 dimensions
            raw: A boolean, which determines if the output image
                 is thresholded
            threshold: An integer, which is the threshold of the back
                       projected image
            disc_kernel: A tuple of 2 integers, which is the size of the
                         kernel for filtering
        return: A numpy ndarray, which has 3 dimensions
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hist,
                                  [0, 180, 0, 256], 1)
        if raw:
            return dst
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, disc_kernel)
        cv2.filter2D(dst, -1, disc, dst)
        thresh = cv2.threshold(dst, threshold, 255, 0)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        return cv2.bitwise_and(image, thresh)


class TemplateMatcher:
    """This class is used to find parts of an image that match a template."""

    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    def __init__(self, template):
        """Initializes the TemplateMatcher by converting and setting the template.
        params:
            template: A numpy ndarray, which has 2 or 3 dimensions (BGR)
        """
        self.template = template
        self.h, self.w = self.template.shape[:2]

    def match_coords(self, image, method=cv2.TM_CCOEFF_NORMED):
        """Finds the top left point and dimensions (width, height)
           of a subimage that most matches the template.
        params:
            image: A numpy ndarray, which has 2 or 3 dimensions (BGR)
            method: A cv2 constant or integer, which determines the
                    method of finding a match
        returns: A tuple of 2 tuples with 2 integers in each
                 ((left, top), (width, height)) and a float
                 of the confidence
        """
        result = cv2.matchTemplate(image, self.template, method)
        min_loc, max_loc = cv2.minMaxLoc(result)[2:]
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        return top_left, (self.w, self.h), result[top_left[::-1]]

    def match_draw_rect(self, image, color=(0, 255, 0), thickness=2,
                        method=cv2.TM_CCOEFF_NORMED):
        """Finds the top left point and dimensions (width, height)
           of a subimage that most matches the template and then
           draws a rectange with those coords.
        params:
            image: A numpy ndarray, which has 2 or 3 dimensions (BGR)
            color: A tuple of 1 or 3 integers, which represents the
                   color of the drawn rectangle
            thickness: An integer, which is the thickness of the
                       rectangle line
            method: A cv2 constant or integer, which determines the
                    method of finding a match
        returns: A float of the confidence
        """
        top_left, (w, h), result = self.match_coords(image, method)
        cv2.rectangle(image, top_left, (top_left[0] + w, top_left[1] + h),
                      color, thickness)
        return result

    def match_draw_all_rects(self, image, threshold=.8, color=(0, 255, 0),
                             thickness=2, method=cv2.TM_CCOEFF_NORMED):
        """Finds the top left point and dimensions (width, height)
           of all subimages that match the template and then
           draws a rectange with those coords.
        params:
            image: A numpy ndarray, which has 2 or 3 dimensions (BGR)
            threshold: A float, which is the threshold for being a match
                       (higher more of a match)
            color: A tuple of 1 or 3 integers, which represents the
                   color of the drawn rectangle
            thickness: An integer, which is the thickness of the
                       rectangle line
            method: A cv2 constant or integer, which determines the
                    method of finding a match
        returns: A tuple of 2 tuples with 2 integers in each
                 ((left, top), (width, height))
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_image, self.template, method)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            ys, xs = np.where(result <= threshold)
        else:
            ys, xs = np.where(result >= threshold)
        for x, y in zip(xs, ys):
            # Not checking for overlaps
            cv2.rectangle(image, (x, y), (x + self.w, y + self.h),
                          color, thickness)


class Camera:
    """This class is used for capturing pictures with the
       computer's camera.
    """
    _DEVICES = set()

    def __init__(self, fps=30, camera_device=0):
        """Initializes the camera and checks if it worked.
        params:
            fps: An integer, which is the number of frames per second
            camera_device: An integer, which determines the device to use
        """
        self.camera_device = camera_device
        self.camera = cv2.VideoCapture(camera_device)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        if not self.camera.isOpened():
            raise Exception("Camera could not be found")
        if camera_device in Camera._DEVICES:
            raise Exception("Camera device already in use")
        else:
            Camera._DEVICES.add(camera_device)

    def __enter__(self):
        if not self.camera.isOpened():
            self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self):
        if self.camera_device in Camera._DEVICES:
            raise Exception("Camera device already in use")
        else:
            Camera._DEVICES.add(self.camera_device)
        if not self.camera.open(self.camera_device):
            raise Exception("Camera could not be found")

    def close(self):
        if self.camera.isOpened() and self.camera_device in Camera._DEVICES:
            Camera._DEVICES.remove(self.camera_device)
        self.camera.release()

    def capture(self, filename=None, target_shape=None, color=True):
        """Uses the camera object to capture an iamge.
        params:
            filename: A string, which is the directory or filename to
                      save image to
            target_shape: A tuple with the vertical size then horizontal
                          size
            color: A boolean, which determines if the image should be
                   converted to gray scale
        return: None or a numpy ndarray, which has 2 or 3 dimensions
        """
        grabbed, frame = self.camera.read()
        if not grabbed:
            return False
        if target_shape is not None:
            frame = resize(frame, target_shape)
        if not color:
            frame = gray(frame)
        if filename is not None:
            cv2.imwrite(filename, frame)
            return True
        else:
            return frame

    def record(self, num_frames=None, filename=None,
               target_shape=None, color=True):
        """Uses the camera object to capture many iamges in a row.
        params:
            num_frmaes: An integer, which is the number of frames to capture
            filename: A string, which is the directory or filename to
                      save image to
            target_shape: A tuple with the vertical size then horizontal
                          size
            color: A boolean, which determines if the image should be
                   converted to gray scale
        return: None or a list of numpy ndarrays, which have 2 or 3 dimensions
        """
        frames = []
        for _ in range(num_frames):
            grabbed, frame = self.camera.read()
            if not grabbed:
                return False
            frames.append(frame)
        for ndx in range(num_frames):
            if target_shape is not None:
                frames[ndx] = resize(frames[ndx], target_shape)
            if not color:
                frames[ndx] = gray(frames[ndx])
                cv2.imwrite(f'{ndx+1}_{filename}', frames[ndx])
                return True
        if filename is None:
            return frames
        else:
            return True


class LockDict:
    """This class is used by camera and is a thread safe dict."""

    def __init__(self, dict_=None):
        """Initializes the LockDict.
        params:
            dict_: A dictionary
        """
        self.dict = dict_ if dict_ else {}
        self.lock = Lock()

    def __getitem__(self, key):
        """Gets an item from a key.
        params:
            key: A hashable value
        return: A value
        """
        self.lock.acquire()
        if key not in self.dict.keys():
            raise KeyError
        x = self.dict[key]
        self.lock.release()
        return x

    def __setitem__(self, key, value):
        """Sets a key to a value.
        params:
            key: A hashable value
            value: A value
        """
        self.lock.acquire()
        self.dict[key] = value
        self.lock.release()

    def __contains__(self, key):
        """Checks if key is in the dictionary.
        params:
            key: A hashable value
        """
        self.lock.acquire()
        x = key in self.dict
        self.lock.release()
        return x

    def __delitem__(self, key):
        """Deletes a key and value.
        params:
            key: A hashable value
        """
        self.lock.acquire()
        del self.dict[key]
        self.lock.release()

    def keys(self):
        """Returns a set of all the keys.
        return: A set
        """
        self.lock.acquire()
        x = set(self.dict.keys())
        self.lock.release()
        return x

    def values(self):
        """Returns a list of all the values.
        return: A list
        """
        self.lock.acquire()
        x = list(self.dict.values())
        self.lock.release()
        return x

    def items(self):
        """Returns a list of all the keys and values.
        return: A list of tuples with key then value
        """
        self.lock.acquire()
        x = list(self.dict.items())
        self.lock.release()
        return x


class Windows:
    """This class is used to displays images."""
    CREATED = False

    def __init__(self, update_delay=1):
        """Initializes the Dictionaries for holding the windows.
           (Can only have one instance per process)
        params:
            update_delay: An integer, which is the number of ms
                          to delay each update (must be > 0)
        """
        if Windows.CREATED:
            raise Exception('Only one Windows instance can exist per process.')
        else:
            Windows.CREATED = True
        assert update_delay > 0, 'update_delay must be greater than 0'
        self.update_delay = update_delay
        self.windows = LockDict()
        self.callbacks = LockDict()
        self.stop_event = Event()
        self.thread = None

    def start(self):
        """Starts the thread for updating."""
        if self.thread is None:
            self.thread = Thread(target=self._update, daemon=True)
            self.thread.start()
        self.stop_event.clear()

    def stop(self):
        """Stops the thread from updating the windows and removes
           all windows.
        """
        if self.thread is not None:
            self.stop_event.set()

    def __enter__(self):
        """Starts the thread for updating."""
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        """Stops the thread from updating the windows and removes
           all windows.
        """
        self.stop()
        if type is not None:
            return False

    def _update(self):
        """Updates the windows. (Called by thread)"""
        windows_open = False
        while True:
            while not self.stop_event.is_set():
                windows_open = True
                for name, image in self.windows.items():
                    if self.callbacks[name] is not None:
                        cv2.namedWindow(name)
                        cv2.setMouseCallback(name, self.callbacks[name])
                        self.callbacks[name] = None
                    cv2.imshow(name, image)
                    cv2.waitKey(self.update_delay)
            if windows_open:
                cv2.destroyAllWindows()
                self.windows = LockDict()
                self.callbacks = LockDict()
                windows_open = False
            sleep(.01)

    def add(self, name='Image', image=None, mouse_callback=None):
        """Adds an image to the update dictionary.
        params:
            name: A string, which is the unguaranteed name of the window.
            image: A numpy ndarray, which has 2 or 3 dimensions
            mouse_callback: A function, which can be called on window events
        return: A string, which is the name for the window
        """
        ndx = 1
        temp_name = name
        while temp_name in self.windows:
            temp_name = f'{name} ({ndx})'
            ndx += 1
        name = temp_name
        if image is None:
            self.windows[name] = np.full((100, 100), 0, dtype=np.uint8)
            self.callbacks[name] = mouse_callback
        else:
            self.windows[name] = image
            self.callbacks[name] = mouse_callback
        return name

    def set(self, name, image):
        """Sets the window to image.
        params:
            name: A string, which is the unguaranteed name of the window.
            image: A numpy ndarray, which has 2 or 3 dimensions
        """
        self.windows[name] = image

    def remove(self, name):
        """Removes a window from the update dictionary.
        params:
            name: A string, which is the unguaranteed name of the window.
        """
        del self.windows[name]
        del self.callbacks[name]
        cv2.destroyWindow(name)

    @staticmethod
    def mouse_callback_logger(event, x, y, flags, param):
        """Logs all the events of a window.
        params:
            event: A cv2 constant or an integer
            x: An integer, which is the horizontal position of the event
            y: An integer, which is the vertical position of the event
            flags: A cv2 constant or an integet
            param: A list of additional variables
        """
        log = []
        if flags == cv2.EVENT_FLAG_LBUTTON:
            log.append('left')
        elif flags == cv2.EVENT_FLAG_RBUTTON:
            log.append('right')
        elif flags == cv2.EVENT_FLAG_MBUTTON:
            log.append('middle')
        elif flags == cv2.EVENT_FLAG_CTRLKEY:
            log.append('CTRL')
        elif flags == cv2.EVENT_FLAG_SHIFTKEY:
            log.append('SHIFT')
        elif flags == cv2.EVENT_FLAG_ALTKEY:
            log.append('ALT')

        if event == cv2.EVENT_MOUSEMOVE:
            log.append('mouse moved')
        elif event == cv2.EVENT_LBUTTONDOWN:
            log.append('left button down')
        elif event == cv2.EVENT_RBUTTONDOWN:
            log.append('right button down')
        elif event == cv2.EVENT_MBUTTONDOWN:
            log.append('middle button down')
        elif event == cv2.EVENT_LBUTTONUP:
            log.append('left button up')
        elif event == cv2.EVENT_RBUTTONUP:
            log.append('right button up')
        elif event == cv2.EVENT_MBUTTONUP:
            log.append('middle button up')
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            log.append('left button double click')
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            log.append('right button double click')
        elif event == cv2.EVENT_MBUTTONDBLCLK:
            log.append('middle button double click')
        elif event == cv2.EVENT_MOUSEWHEEL:
            log.append('mouse wheel')
        elif event == cv2.EVENT_MOUSEHWHEEL:
            log.append('mouse horizontal wheel')

        log.append(f'x: {x} y: {y}')

        print(' + '.join(log))


if __name__ == '__main__':
    def avg_time(func, args=None, loops=1000):
        if args is None:
            start_time = time()
            for _ in range(loops):
                result = func()
            end_time = time()
        else:
            start_time = time()
            for _ in range(loops):
                result = func(*args)
            end_time = time()
        print((end_time - start_time) / loops)

    c = Camera()
    x = c.capture()
    print(gray(x).shape)
    sleep(1)
    x2 = c.capture()
    ws = Windows()
    try:
        ws.start()
        ws.add('x', x)
        ws.add('gray', gray(x))
        ws.add('resize', resize(x, (200, 200)))
        ws.add('pyr up', pyr(x, 2))
        ws.add('pyr down', pyr(x, -2))
        ws.add('increase brightness', increase_brightness(x, 10))
        ws.add('increase brightness (relative)',
               increase_brightness(x, -10, True))
        ws.add('set brightness', set_brightness(x, 10))
        ws.add('set brightness (relative)', set_brightness(x, 50, True))
        ws.add('set gamma', set_gamma(x, 1.5))
        ws.add('clahe', apply_clahe(x))
        ws.add('clahe gray', apply_clahe(gray(x)))
        ws.add('equalize', equalize(x))
        ws.add('equalize gray', equalize(gray(x)))
        ws.add('rotate', rotate(x, 45))
        ws.add('translate', translate(x, 10, 20))
        ws.add('crop rect', crop_rect(x, 100, 50, 100, 100))
        ws.add('crop rect coords', crop_rect_coords(x, 100, 100, 200, 200))
        ws.add('crop', crop(x, (300, 300), 200, 0))
        ws.add('shrink sides', shrink_sides(x, 100, 100, 100, 100))
        ws.add('zoom', zoom(x, (240, 320), 0, 0))
        ws.add('pad', pad(x, 50, 50, 50, 50))
        ws.add('blend', blend(x, x2, .7))
        lb, ub = compute_color_ranges([x, x2], percentage_captured=50)
        ws.add('mask', create_mask_of_colors_in_range(x, lb, ub))
        ws.add('magnitude spectrum', np.uint8(create_magnitude_spectrum(x)))
        ws.add('high pass filter', freq_filter_image(x))
        ws.add('low pass filter', freq_filter_image(x, False))
        tm = TemplateMatcher(crop_rect(x, 100, 50, 100, 100))
        # tm.match_draw_rect(x)
        tm.match_draw_all_rects(x)
        ws.add('match', x)
        input('take pic')
        x3 = c.record(100)
        hbp = HistogramBackProjector(x3[0])
        ws.add('hist backproject', np.vstack((hbp.backproject(x), x3[0])))
        x4 = [bgr2hsv(cv2.GaussianBlur(y, (9, 9), 0)) for y in x3]
        hist = create_histograms(x4, hsv_images=True, channels=[0, 1])
        plt.imshow(hist, interpolation='nearest')
        plt.show()
        sleep(1000)
    finally:
        ws.stop()
    c.close()
