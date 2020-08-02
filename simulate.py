import os
import sys
sys.path.append("sd/")
import datetime as dt
import argparse
from dateutil import parser as dparser

import analysis as ala
import plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cs", "--case", type=int, default=0, help="Different cases (0-10)")
    parser.add_argument("-sn", "--stn", default="ott", help="Station code (default ott)")
    parser.add_argument("-rd", "--rad", action="store_true", help="Is radar (default False)")
    parser.add_argument("-ev", "--event", default=dt.datetime(2015,3,11,16,22), help="Start date (default 2015-3-11T16:22)",
            type=dparser.isoparse)
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,11,16,15), help="Start date (default 2015-3-11T16:05)",
            type=dparser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,11,16,24), help="End date (default 2015-3-11T16:25)",
            type=dparser.isoparse)
    parser.add_argument("-c", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    parser.add_argument("-ac", "--acase", type=int, default=0, help="Different analysis case cases (0-10)")
    args = parser.parse_args()
    if args.verbose: print("\n Parameter list for simulation ")
    for k in vars(args).keys():
        if args.verbose: print("     " , k , "->" , str(vars(args)[k]))
    if args.case == 0: plot.example_riom_plot()
    if args.case == 1: plot.example_rad_plot()
    if args.case == 2: plot.example_hrx_plot()
    if args.case == 3: os.system("cp ../_images/instruments.png images/fov.png")
    if args.case == 4: plot.Statistics(args)._fig_()
    if args.case == 5: ala.RioStats(args)._exe_()
    if args.case == 6: ala.SDStats(args)._exe_()

    os.system("rm *.log")
    os.system("rm -rf __pycache__/")
