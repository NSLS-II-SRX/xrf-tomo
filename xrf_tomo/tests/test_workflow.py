import numpy as np
import pytest

from xrf_tomo.xrf_tomo_workflow import _shift_images

_image_shift_images = np.zeros([5, 7])
_image_shift_images[1:4, 2:5] = 1

_image_shift_images_exp_1 = np.zeros([5, 7])
_image_shift_images_exp_1[0:3, 2:5] = 1

_image_shift_images_exp_2 = np.zeros([5, 7])
_image_shift_images_exp_2[2:5, 2:5] = 1

_image_shift_images_exp_3 = np.zeros([5, 7])
_image_shift_images_exp_3[1:4, 1:4] = 1

_image_shift_images_exp_4 = np.zeros([5, 7])
_image_shift_images_exp_4[1:4, 3:6] = 1

_image_shift_images_exp_5 = np.zeros([5, 7])
_image_shift_images_exp_5[2:5, 1:4] = 1

_image_shift_images_exp_6 = np.zeros([5, 7])
_image_shift_images_exp_6[0:3, 3:6] = 1


# fmt: off
@pytest.mark.parametrize("dx, dy, expected_result", [
    (0, 0, _image_shift_images),
    (0, 1, _image_shift_images_exp_1),  # Vertical shift
    (0, -1, _image_shift_images_exp_2),
    (1, 0, _image_shift_images_exp_3),  # Horizontal shift
    (-1, 0, _image_shift_images_exp_4),
    (1, -1, _image_shift_images_exp_5),  # Vertical and horizontal shift
    (-1, 1, _image_shift_images_exp_6),
])
# fmt: on
def test_shift_images(dx, dy, expected_result):
    """
    Tests for ``_shift_images``, which is a trivial function, but it will be used in other tests
    to generate test datasets.
    """
    n_stack = 2
    im_source = np.zeros([n_stack, *_image_shift_images.shape])
    for n in range(n_stack):
        im_source[n, :, :] = _image_shift_images
    dx_array = np.array([dx] * n_stack)
    dy_array = np.array([dy] * n_stack)

    im_shifted = _shift_images(im_source, dx=dx_array, dy=dy_array)

    for n in range(n_stack):
        assert np.max(np.abs(im_shifted[n] - expected_result)) < 0.1
