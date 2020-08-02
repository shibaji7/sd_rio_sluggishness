import os
import datetime as dt
import pandas as pd
from scipy import stats, signal
import numpy as np

from pysolar.solar import get_altitude

import statsmodels.api as sm
from statsmodels.formula.api import ols

class GOES(object):
    def __init__(self, edate, _sdate=None, _edate=None, g=15, sec=2):
        self.g = g
        self.edate = edate
        if _sdate == None: _sdate = edate - dt.timedelta(minutes=20)
        if _edate == None: _edate = edate + dt.timedelta(minutes=40)
        self._sdate = _sdate
        self._edate = _edate
        self.url = "https://satdat.ngdc.noaa.gov/sem/goes/data/full/%4d/%02d/goes%02d/netcdf\
                /g%02d_xrs_{sec}s_%s_%s.nc".replace("{sec}",str(sec))
        self.tmp = "/tmp/g%02d_xrs_1m_%s_%s.nc"
        self._fname = "/tmp/%s.csv"%edate.strftime("%Y-%m-%d-%H-%M")
        self.save = "slug/goes/%s.csv"%edate.strftime("%Y-%m-%d-%H-%M")
        return
    
    def delete(self,dfiles):
        for d in dfiles:
            if os.path.exists(d): os.remove(d)
        return
    
    def to_remote(self):
        os.system("scp %s shibaji7@newriver1.arc.vt.edu:/home/shibaji7/%s"%(self._fname, self.save))
        return
    
    def from_remote(self):
        os.system("scp shibaji7@newriver1.arc.vt.edu:/home/shibaji7/%s %s"%(self.save, self._fname))
        return
    
    def fetch(self):
        self.from_remote()
        if os.path.exists(self._fname):
            _o = pd.read_csv(self._fname, parse_dates=["times"])
            _o = _o[(_o.times >= self._sdate) & (_o.times <= self._edate)]
        else: _o = self.__download()
        if os.path.exists(self._fname): os.remove(self._fname)
        return _o

class Riometer(object):
    def __init__(self, edate, rio, case=0, _sdate=None, _edate=None):
        self.rio = rio
        if rio != "ott":
            self.riom_code_map = {"talo":"tal","rank":"ran", "pina":"pin", "gill":"gil", "daws":"daw","fsim":"sim","fsmi":"smi",
                    "rabb":"rab","isll":"isl","mcmu":"mcm"}
            self.fmt_map = {0:"norstar_k2_rio-{s}_{d}_v01.txt", 1:"{s}_rio_{d}_v1a.txt"}
            self.fmt = self.fmt_map[case]
            self.case = case
            self.old_rio = self.riom_code_map[rio]
            self.url = "http://data.phys.ucalgary.ca/sort_by_project/GO-Canada/GO-Rio/txt/%d/%02d/%02d/"
            self.tmp = "/tmp/"+self.fmt
            self._fname = "/tmp/%s_%s.csv"%(edate.strftime("%Y-%m-%d-%H-%M"), rio)
            self.save = "slug/riometer/%s_%s.csv"%(edate.strftime("%Y-%m-%d-%H-%M"), rio)
        self.edate = edate
        if _sdate == None: _sdate = edate - dt.timedelta(minutes=20)
        if _edate == None: _edate = edate + dt.timedelta(minutes=40)
        self._sdate = _sdate
        self._edate = _edate
        return

    def delete(self,dfiles):
        for d in dfiles:
            if os.path.exists(d): os.remove(d)
        return
    
    def to_remote(self):
        os.system("scp %s shibaji7@newriver1.arc.vt.edu:/home/shibaji7/%s"%(self._fname, self.save))
        return
    
    def from_remote(self):
        os.system("scp shibaji7@newriver1.arc.vt.edu:/home/shibaji7/%s %s"%(self.save, self._fname))
        return
    
    def fetch(self):
        if self.rio!="ott":
            self.from_remote()
            if os.path.exists(self._fname): _o = pd.read_csv(self._fname, parse_dates=["times"])
            else: _o = self.__download()
            if os.path.exists(self._fname): os.remove(self._fname)
        else:
            _o = pd.DataFrame()
            dirc="/home/shibaji/cleaned_riometer_data/"
            _u = pd.read_csv(dirc + "ott%s_03.csv"%self.edate.strftime("%Y%m%d"))
            _u = _u[_u.Flag==1]
            _o["times"] = pd.to_datetime(_u.Time)
            _o["volt"] = _u.Volts
            _o["absorption"] = _u.Absorp
            _o = _o[(_o.times >= self._sdate) & (_o.times <= self._edate)]
        return _o

def get_riom_loc(stn):
    """ This method is to get the location of the riometer """
    _o = pd.read_csv("csv/riometer.csv")
    _o = _o[_o.rio==stn]
    lat, lon = _o["lat"].tolist()[0], np.mod( (_o["lon"].tolist()[0] + 180), 360 ) - 180
    return lat, lon

def get_radar_loc(stn):
    _o = pd.read_csv("csv/radar.csv")
    _o = _o[_o.rad==stn]
    lat, lon = _o["lat"].tolist()[0], np.mod( (_o["lon"].tolist()[0] + 180), 360 ) - 180
    return lat, lon

def get_sza(lat, lon, d):
    d = d.replace(tzinfo=dt.timezone.utc)
    sza = 90. - get_altitude(lat, lon, d)
    return sza

def slope_analysis(dat, times, sec=2, xT=60):
    t, c = [],[]
    for i in range(len(dat)-xT):
        x = dat[i:i+xT]
        _s, _, _, _, _ = stats.linregress(np.arange(xT)*sec, x)
        c.append(_s)
        t.append(times[i].to_pydatetime() + dt.timedelta(seconds=xT/2))
    ftime = t[c.index(max(c))]
    return ftime

def smooth(x,window_len=51,window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y

def anova(args):
    if args.acase == 0 and not args.rad: fname = "csv/rio_c0.csv"
    if args.acase == 1 and not args.rad: fname = "csv/rio_c1.csv"
    if args.acase == 2 and not args.rad: fname = "csv/rio_c2.csv"
    if args.acase == 1 and args.rad: fname = "csv/rad_c1.csv"
    if args.acase == 2 and args.rad: fname = "csv/rad_c2.csv"
    dat = pd.read_csv(fname)
    response = []
    for llim, ulim in zip([1e-7, 1e-5, 1e-4], [1e-5, 1e-4, 1e-2]):
        df = dat[(dat.fmax > llim) & (dat.fmax < ulim)]
        df["cossza"] = df.sza.transform(lambda x: np.cos(np.deg2rad(x)))
        df["logfmax"] = df.fmax.transform(lambda x: np.log10(x))
        models = {}
        models["m1"] = sm.GLM(df.dt, df[["cossza","lat","logfmax","lt"]].values, family=sm.families.NegativeBinomial())
        for k in models.keys():
            res = models[k].fit()
            response.append(res)
            print(res.summary())
    return response
