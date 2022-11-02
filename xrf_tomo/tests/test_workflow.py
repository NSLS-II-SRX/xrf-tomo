import h5py
import numpy as np
import numpy.testing as npt
import os
import pytest

from xrf_tomo.xrf_tomo_workflow import _shift_images, shift_projections, normalize_pixel_range  # , find_alignment

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
            elif k == "theta":
                f.create_dataset("/exchange/theta", data=v)
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
    imgs = np.zeros([len(shifts), 1, *img.shape])  # Only ONE element
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


@pytest.fixture
def change_test_dir(tmpdir):
    wd = os.getcwd()
    os.chdir(tmpdir)
    yield tmpdir
    os.chdir(wd)


def test_normalize_pixel_range_01(change_test_dir):
    """
    'normalize_pixel_range_01'
    """
    tmpdir = change_test_dir

    v_min, v_max = 0.7, 5.6
    n_proj = 10
    dx = np.zeros([n_proj])
    dy = np.zeros([n_proj])

    elements = ["total_cnt", "Ca_K"]
    n_el = len(elements)

    # Generate the stack of images
    img = np.array(_image_shift_images) * (v_max - v_min) + v_min
    imgs = np.zeros([n_proj, n_el, *img.shape])  # Only ONE element
    for n in range(n_proj):
        imgs[n] = np.copy(img)

    params = {}
    params["recon_proj"] = imgs
    params["recon_del_x"] = dx
    params["recon_del_y"] = dy
    params["elements"] = ["total_cnt", "Ca_K"]

    n_el = len(params["elements"])

    fn_hdf = os.path.join(tmpdir, "single_hdf.h5")
    _create_test_hdf(fn_hdf, params=params)

    with h5py.File(fn_hdf, "r") as f:
        npt.assert_almost_equal(f["reconstruction"]["recon"]["proj"], imgs)
        npt.assert_almost_equal(f["reconstruction"]["recon"]["del_x"], dx)
        npt.assert_almost_equal(f["reconstruction"]["recon"]["del_y"], dy)

    with h5py.File(fn_hdf, "r") as f:
        proj = np.copy(f["/reconstruction/recon/proj"])

    assert proj.shape[0] == n_proj, proj.shape
    assert proj.shape[1] == n_el, proj.shape
    npt.assert_almost_equal(np.min(proj), v_min)
    npt.assert_almost_equal(np.max(proj), v_max)

    proj_norm = normalize_pixel_range("single_hdf.h5", path=tmpdir, read_only=True)

    assert proj_norm.shape[0] == n_proj, proj_norm.shape
    assert proj_norm.shape[1] == n_el, proj_norm.shape
    npt.assert_almost_equal(np.min(proj_norm), 0.0)
    npt.assert_almost_equal(np.max(proj_norm), 1.0)

    normalize_pixel_range("single_hdf.h5", path=tmpdir, read_only=False)

    with h5py.File(fn_hdf, "r") as f:
        proj_norm2 = np.copy(f["/reconstruction/recon/proj"])

    assert proj_norm2.shape[0] == n_proj, proj_norm2.shape
    assert proj_norm2.shape[1] == n_el, proj_norm2.shape
    npt.assert_almost_equal(np.min(proj_norm2), 0.0)
    npt.assert_almost_equal(np.max(proj_norm2), 1.0)

    npt.assert_almost_equal(proj_norm2, proj_norm)

    for n in range(n_el):
        for m in range(n_proj):
            npt.assert_almost_equal(np.min(proj_norm2[m, n, :, :]), 0.0)
            npt.assert_almost_equal(np.max(proj_norm2[m, n, :, :]), 1.0)


# def test_alignment_01(change_test_dir):

#     tmpdir = change_test_dir

#     rot_center = 350
#     radius = 150
#     width = rot_center * 2 + 1

#     # _image_alignment = np.zeros([51, width])
#     _image_alignment = np.zeros([1, width])
#     for n in range(width):
#         d = n - rot_center
#         if abs(d) <= radius:
#             v = 2 * np.sqrt(radius ** 2 - d ** 2)
#             # _image_alignment[20:31, n] = v
#             _image_alignment[0, n] = v

#     shift = (0, 0)

#     theta = np.array(range(180), dtype=np.float32)
#     n_proj = len(theta)

#     n_proj_shifted = -2
#     dx, dy = np.zeros([n_proj]), np.zeros([n_proj])
#     dx[n_proj_shifted] = shift[0]
#     dy[n_proj_shifted] = shift[1]

#     print(f"image_alignment={_image_alignment}")

#     # Generate the stack of images
#     img = np.array(_image_alignment)
#     imgs = np.zeros([n_proj, 1, *img.shape])  # Only ONE element
#     for n in range(n_proj):
#         imgs[n] = np.copy(img)
#     imgs_shifted = np.copy(imgs)
#     # imgs_shifted = np.zeros(imgs.shape)
#     # imgs_shifted[:, 0, :, :] = _shift_images(imgs[:, 0, :, :], dx=-dx, dy=-dy)

#     params = {}
#     params["recon_proj"] = imgs_shifted
#     params["elements"] = ["total_cnt"]
#     params["theta"] = theta

#     fn_hdf = os.path.join(tmpdir, "single_hdf.h5")
#     _create_test_hdf(fn_hdf, params=params)

#     find_alignment(
#         fn_hdf, "total_cnt", iters=10, algorithm="gridrec",
#         center=None, alignment_algorithm="align_seq", save=True
#     )
#     shift_projections(fn_hdf, read_only=False)

#     with h5py.File(fn_hdf, "r") as f:
#         print(f'{np.array(f["reconstruction"]["recon"]["proj"])[n_proj_shifted, 0, :, :]}')
#         print(f"{imgs[n_proj_shifted, 0, :, :]}")
#         print(f'dx={np.array(f["reconstruction"]["recon"]["del_x"])}')
#         print(f'dy={np.array(f["reconstruction"]["recon"]["del_y"])}')
#         npt.assert_almost_equal(f["reconstruction"]["recon"]["proj"], imgs)

#     # assert False
