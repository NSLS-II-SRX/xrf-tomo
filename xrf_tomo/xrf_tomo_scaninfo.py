# flake8: noqa

import numpy as np
import time as ttime
from pyxrf.api_dev import db


def get_tomo_information(start, stop, fn="tomo_info.dat"):

    if not db:
        raise RuntimeError("Databroker is not available")

    # Make a list of scan IDs
    ids = list(np.arange(start, stop + 1))

    header = "Start Time,Scan ID,Theta,Use,Filename,X Start,X Stop,Num X,Y Start,Y Stop,Num Y,Dwell,UID,Status\n"

    with open(fn, "a") as f:
        f.write(header)
        ind = 0
        for scanid in ids:
            h = db[int(scanid)]
            textout = ""
            textout = ttime.ctime(h.start["time"])
            textout = textout + "," + str(h.start["scan_id"])
            # textout = textout + ',' + str(h.start['theta'])
            textout = textout + "," + str(th[ind])
            textout = textout + ",x"
            textout = textout + "," + f"scan2D_{scanid}_xs_sum4ch.h5"
            textout = textout + "," + str(h.start["scan_input"])[1:-1].replace(" ", "")
            textout = textout + "," + h.start["uid"]
            textout = textout + "," + h.stop["exit_status"] + "\n"
            f.write(textout)
            if h.stop["exit_status"] == "success":
                ind = ind + 1
