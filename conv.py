import aacgmv2
import datetime as dt
import numpy as np
import pandas as pd

np.set_printoptions(formatter={"float_kind": lambda x:"{:.2f}".format(x)})
dtime = dt.datetime(2013, 11, 3)
x = pd.read_csv("csv/radar.csv")

for rad, lat, lon in zip(x.rad, x.lat, x.lon):
    print("Rad(%s)"%rad, np.array(aacgmv2.get_aacgm_coord(lat, lon, 300, dtime)))

y = pd.read_csv("csv/riometer_details.csv")
for rio, lat, lon in zip(y.RIO, y.LAT, y.LON):
    print("Rio(%s)"%rio, (lat, np.round(np.mod( (lon + 180), 360 ) - 180, 2)), np.array(aacgmv2.get_aacgm_coord(lat, lon, 300, dtime)))
