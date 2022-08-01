import datetime as dt
import sys

sys.path.append("HAO.LASP/")


if __name__ == "__main__":
    from get_flare_data import GOES
    from fetch_riometer_data import get_riometer_zipped_data

    # Fetch GOES low and high-reolution data
    g = GOES.FetchGOES(
        "data/2015-03-11-16-20/",
        [
            dt.datetime(2015, 3, 11, 16),
            dt.datetime(2015, 3, 11, 16, 30),
        ],
    )
    
    # Fetch riometer observations
    r = get_riometer_zipped_data(
            "data/2015-03-11-16-20/",
            dt.datetime(2015, 3, 11),
            "ott"
    )
    r = r[
        (r.date>=dt.datetime(2015, 3, 11, 16, 10)) & 
        (r.date<=dt.datetime(2015, 3, 11, 16, 30))
    ]
    r = r[
        (r.alpha<=3) & (r.alpha>=-.5)
    ]
    
    g.plot_TS_dataset_compare(
        ref_data = {
            "x": r.date.tolist(),
            "y": r.alpha.tolist(),
            "ylabel": r"Absorption $(\beta)$, dB",
            "ylim": [0, 3]
        },
        xlim = [
            dt.datetime(2015, 3, 11, 16, 10),
            dt.datetime(2015, 3, 11, 16, 30)
        ],
        fname="data/2015-03-11-16-20/goes.png"
    )
    
    