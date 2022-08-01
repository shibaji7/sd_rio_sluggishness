import os
import pandas as pd

def get_riometer_zipped_data(folder, date, rio, ver="03"):
    fname = "{rio}{dx}_{ver}.txt".format(rio=rio, dx=date.strftime("%Y%m%d"), ver=ver)
    if not os.path.exists(folder + "/" + fname):
        base = "LFS/LFS_riometer_data/proc/riometer/{year}/".format(year=date.year)
        gzfname = "{rio}{dx}_{ver}.txt.gz".format(rio=rio, dx=date.strftime("%Y%m%d"), ver=ver)
        fpath = base + gzfname
        cmd = "scp shibaji7@cascades1.arc.vt.edu:~/{fpath} {folder}".format(
            fpath=fpath, 
            folder=folder
        )
        os.system(cmd)
        cmd = "gzip -d {folder}/{gzfname}".format(
            folder=folder,
            gzfname=gzfname
        )
        os.system(cmd)
    r = pd.DataFrame()
    with open(folder + "/" + fname, "r") as file:
        lines = file.readlines()
        o = []
        for l in lines[17:-1]:
            l = list(filter(None, l.replace("\n", "").split(" ")))
            o.append(
                {
                    "date": date.replace(hour=int(l[0]), minute=int(l[1]), second=int(l[2])),
                    "alpha": float(l[-2]),
                    "flag": int(l[-1])
                }
            )
        r = pd.DataFrame.from_records(o)
    return r