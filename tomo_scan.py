# flake8: noqa

# Run XRF-tomography
#
import numpy as np
from scipy.ndimage.measurements import center_of_mass


def haz_angles(a, b, n):
    th = np.linspace(a, b, num=n)
    th2 = np.concatenate((th[::2], th[-2::-2]))
    return th2


# Define a function to call from the RunEngine
def run_xrftomo(x0, x1, nx, y0, y1, ny, ct, th=None, th_offset=45):
    # x0 = x starting point
    # x1 = x finish point
    # nx = number of points in x
    # y0 = y starting point
    # y1 = y finish point
    # ny = number of points in y
    # th = angles to scan at

    # Set the angles for collection
    if th is None:
        th = np.linspace(0, 180, 181)
    th = th + th_offset

    # Step scan parameters
    xnumstep = nx - 1
    xstepsize = (x1 - x0) / xnumstep
    ynumstep = ny - 1
    ystepsize = (y1 - y0) / ynumstep

    # Run the scan
    for i in th:
        print("Scanning at: %f" % (i))
        # Rotate the sample
        # hf_stage.th.move(i, wait=True)
        yield from mv(hf_stage.th, i)

        # Run the scan
        # yield from y_scan_and_fly(y0, y1, ny, x0, x1, nx, ct)
        yield from hf2dxrf(
            xstart=x0,
            xnumstep=xnumstep,
            xstepsize=xstepsize,
            ystart=y0,
            ynumstep=ynumstep,
            ystepsize=ystepsize,
            acqtime=ct,
        )


# Define a function to call from the RunEngine
def fly_xrftomo(x0, x1, nx, y0, y1, ny, ct, th=None, th_offset=0):
    # x0 = x starting point
    # x1 = x finish point
    # nx = number of points in x
    # y0 = y starting point
    # y1 = y finish point
    # ny = number of points in y
    # th = angles to scan at

    # Set the angles for collection
    if th is None:
        th = np.linspace(0, 180, 181)
    th = th + th_offset

    # Step scan parameters
    # xnumstep = nx - 1
    # xstepsize = (x1 - x0) / xnumstep
    # ynumstep = ny - 1
    # ystepsize = (y1 - y0) / ynumstep

    # Run the scan
    for i in th:
        print("Scanning at: %f" % (i))
        # Rotate the sample
        # hf_stage.th.move(i, wait=True)
        yield from mv(hf_stage.th, i)

        # Run the scan
        yield from scan_and_fly(x0, x1, nx, y0, y1, ny, ct)
        # yield from hf2dxrf(xstart=x0, xnumstep=xnumstep, xstepsize=xstepsize,
        #                    ystart=y0, ynumstep=ynumstep, ystepsize=ystepsize,
        #                    acqtime=ct)


# Define a function to call from the RunEngine
def fly_xrftomo3(x0, x1, nx, y0, y1, ny, ct, th=None, th_offset=0, centering_method="com", extra_dets=[]):
    # x0 = x starting point
    # x1 = x finish point
    # nx = number of points in x
    # y0 = y starting point
    # y1 = y finish point
    # ny = number of points in y
    # th = angles to scan at
    # th_offset = offset value to relate to rotation stage
    # centering_method = method used to account for sample motion and center
    #                    the sample
    #                    'none' = no correction
    #                    'com'  = center of mass

    # Set the angles for collection
    if th is None:
        th = np.linspace(0, 180, 181)
    th = th + th_offset

    # Open the shutter
    yield from mv(shut_b, "Open")

    # Run the scan
    for i in th:
        print("Scanning at: %f" % (i))
        # Rotate the sample
        yield from mv(nano_stage.th, i)
        yield from bps.sleep(1)
        # TH_THRESHOLD = 0.1
        # while True:
        #     yield from mv(hf_stage.th, i)
        #     yield from bps.sleep(1)
        #     # Check if it actually moved
        #     th_SP = nano_stage.th.read()['hf_stage_th_user_setpoint']['value']
        #     th_RBV = nano_stage.th.read()['hf_stage_th']['value']
        #     if (np.abs(th_SP - th_RBV) > TH_THRESHOLD):
        #         print('Reactuating rotation stage...')
        #         if (th_RBV - th_SP) < 0:
        #             yield from mv(nano_stage.th, i-5)
        #             yield from bps.sleep(3)
        #             yield from mv(nano_stage.th, i)
        #         else:
        #             yield from mv(nano_stage.th, i+5)
        #             yield from bps.sleep(3)
        #             yield from mv(nano_stage.th, i)
        #     else:
        #         break

        # Run the scan
        # uid = yield from scan_and_fly(x0, x1, nx, y0, y1, ny, ct)
        scan_flag = True
        # yield from scan_and_fly(x0, x1, nx, y0, y1, ny, ct, extra_dets=extra_dets)
        while scan_flag:
            try:
                yield from nano_scan_and_fly(x0, x1, nx, y0, y1, ny, ct, extra_dets=extra_dets, shutter=False)
                # yield from y_scan_and_fly(y0, y1, ny, x0, x1, nx, ct, extra_dets=extra_dets, shutter=False)
                scan_flag = False
            # except add exception for control -c
            except Exception as e:
                print(e)
                print("\nProblem was detected.\nWaiting 60 seconds and trying again...\n")
                yield from bps.sleep(60)

        # Center the sample
        if centering_method == "com":
            print("Centering sample using center of mass...")
            # Get the data
            flag_get_data = True
            while flag_get_data:
                # h = db[uid]
                h = db[-1]
                try:
                    d = list(h.data("fluor", stream_name="stream0", fill=True))
                    d = np.array(d)
                    d_I0 = list(h.data("i0", stream_name="stream0", fill=True))
                    d_I0 = np.array(d_I0)
                    x = list(h.data("enc1", stream_name="stream0", fill=True))
                    y = list(h.data("enc2", stream_name="stream0", fill=True))
                    flag_get_data = False
                except:
                    yield from bps.sleep(1)

            # NEED TO FIX
            # Mn K-a1: 5900
            # Co K-a1: 6931
            # Ni K-a1: 7480
            d = np.sum(d[:, :, :, 580:758], axis=(2, 3))
            d = d / d_I0
            d = d.T

            # Calculate center of mass
            # (com_y, com_x)  = center_of_mass(d)  # for y scans
            (com_x, com_y) = center_of_mass(d)  # for flying x scans
            # need to rewrite with (x1-x0)/nx...
            com_x = x[0][0] + com_x * (x[0][1] - x[0][0])
            com_y = y[0][0] + com_y * (y[1][0] - y[0][0])
            # print(f'Center of mass X: {com_x}')

            # Calculate new center
            extent = x1 - x0
            old_center = x0 + 0.5 * extent
            dx = old_center - com_x
            extent_y = y1 - y0
            old_center_y = y0 + 0.5 * extent_y
            dy = old_center_y - com_y

            # Check new location
            THRESHOLD = 0.50 * extent
            if np.isfinite(com_x) is False:
                print("Center of mass is not finite!")
                new_center = old_center
            elif np.abs(dx) > THRESHOLD:
                print("New scan center above threshold")
                new_center = old_center
            else:
                new_center = com_x
            x0 = new_center - 0.5 * extent
            x1 = new_center + 0.5 * extent
            print(f"Old center: {old_center}")
            print(f"New center: {new_center}")
            print(f"Difference: {dx}")

            THRESHOLD = 0.50 * extent_y
            if np.isfinite(com_y) is False:
                print("Center of mass is not finite!")
                new_center_y = old_center_y
            elif np.abs(dy) > THRESHOLD:
                print("New scan center above threshold")
                new_center_y = old_center_y
            else:
                new_center_y = com_y
            y0 = new_center_y - 0.5 * extent_y
            y1 = new_center_y + 0.5 * extent_y
            print(f"Old center: {old_center_y}")
            print(f"New center: {new_center_y}")
            print(f"Difference: {dy}")

    # Close the shutter
    yield from mv(shut_b, "Close")
