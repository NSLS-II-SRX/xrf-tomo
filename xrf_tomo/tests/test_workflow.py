import h5py
import numpy as np
import numpy.testing as npt
import os
import pytest

from xrf_tomo.xrf_tomo_workflow import _shift_images, shift_projections

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
def test_shift_images_01(dx, dy, expected_result):
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


def _create_test_hdf(fn, *, params):
    with h5py.File(fn, "w") as f:
        f.create_group("exchange")
        f.create_group("measurement")
        f.create_group("instrument")
        f.create_group("provenance")
        f.create_group("reconstruction")
        f.create_group("reconstruction/fitting")
        f.create_group("reconstruction/recon")

        for k, v in params.items():
            if k == "recon_proj":
                f.create_dataset("reconstruction/recon/proj", data=v, compression="gzip")
            elif k == "recon_del_x":
                f.create_dataset("reconstruction/recon/del_x", data=v)
            elif k == "recon_del_y":
                f.create_dataset("reconstruction/recon/del_y", data=v)
            elif k == "elements":
                elements = np.array([_.encode() for _ in v])
                f.create_dataset("/reconstruction/fitting/elements", data=elements)
            else:
                assert False, f"Unknown keyword {k!r}"


def test_shift_projections_01(tmpdir):
    """
    ``shift_projection``: basic test
    """
    shifts = [(0, 0), (-1, 1), (1, 0), (0, 1), (1, -1)]

    # Arrays of shift values
    dx = np.array([_[0] for _ in shifts])
    dy = np.array([_[1] for _ in shifts])

    # Generate the stack of images
    img = np.array(_image_shift_images)
    imgs = np.zeros([len(shifts), 1, *img.shape])
    for n in range(len(shifts)):
        imgs[n] = np.copy(img)
    imgs_shifted = np.zeros(imgs.shape)
    imgs_shifted[:, 0, :, :] = _shift_images(imgs[:, 0, :, :], dx=-dx, dy=-dy)

    params = {}
    params["recon_proj"] = imgs_shifted
    params["recon_del_x"] = dx
    params["recon_del_y"] = dy
    params["elements"] = ["total_cnt"]

    fn_hdf = os.path.join(tmpdir, "single_hdf.h5")
    _create_test_hdf(fn_hdf, params=params)

    with h5py.File(fn_hdf, "r") as f:
        npt.assert_almost_equal(f["reconstruction"]["recon"]["proj"], imgs_shifted)
        npt.assert_almost_equal(f["reconstruction"]["recon"]["del_x"], dx)
        npt.assert_almost_equal(f["reconstruction"]["recon"]["del_y"], dy)

    shift_projections(fn_hdf, read_only=False)

    with h5py.File(fn_hdf, "r") as f:
        print(f'{np.array(f["reconstruction"]["recon"]["proj"])}')
        print(f"{imgs}")
        npt.assert_almost_equal(f["reconstruction"]["recon"]["proj"], imgs)


def test_alignment_01(tmpdir):
    pass
