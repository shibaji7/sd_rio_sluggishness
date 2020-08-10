import os
import datetime as dt
import pandas as pd
from scipy import stats, signal
import numpy as np
from timezonefinder import TimezoneFinder
from dateutil import tz
from scipy import signal
import traceback
from PyIF import te_compute as te

from pysolar.solar import get_altitude

import statsmodels.api as sm
from statsmodels.formula.api import ols

import plot
import get_sd_data as gsd

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

    @staticmethod
    def get(dn, start=None, end=None, g=15, sec=2):
        o = GOES(dn, start, end, g, sec).fetch()
        return o

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
            if os.path.exists(self._fname):
                _o = pd.read_csv(self._fname, parse_dates=["times"])
                os.remove(self._fname)
            else: _o = pd.DataFrame()
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

    @staticmethod
    def get(dn, stn, start=None, end=None):
        o = Riometer(dn, stn, case=0, _sdate=start, _edate=end).fetch()
        return o

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
    if len(t) > 0: ftime = t[c.index(max(c))]
    else: ftime = None
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
            print(res.summary().as_text())
    return response

def anova_ml(args):
    if args.acase == 1 and not args.rad: fname = "csv/rio_c1.csv"
    if args.acase == 2 and not args.rad: fname = "csv/rio_c2.csv"
    if args.acase == 1 and args.rad: fname = "csv/rad_c1.csv"
    if args.acase == 2 and args.rad: fname = "csv/rad_c2.csv"
    df = pd.read_csv(fname)
    df.dt = np.abs(df.dt)
    if args.acase == 1 and not args.rad: df = df[(df.dt>20) & (df.sza<140) & (df.sza>60)]
    if args.acase == 2 and args.rad: df = df[(df.dt!=0) & (df.dt!=360)]
    df["cossza"] = df.sza.transform(lambda x: np.cos(np.deg2rad(x)))
    df["logfmax"] = df.fmax.transform(lambda x: np.log10(x))
    models = {}
    models["m1"] = sm.GLM(df.dt, df[["cossza","lat","logfmax","lt"]].values, family=sm.families.NegativeBinomial())
    response = []
    for k in models.keys():
        res = models[k].fit()
        response.append(res)
        print(res.summary().as_text())
    return response

def get_LT(lat, lon, d):
    tf = TimezoneFinder()
    from_zone = tz.tzutc()
    to_zone = tz.gettz(tf.timezone_at(lng=lon, lat=lat))
    x = d.replace(tzinfo=from_zone).astimezone(to_zone).to_pydatetime()
    now = x.replace(hour=0,minute=0,second=0)
    LT = (x - now).total_seconds()/3600.
    return LT

class RioStats(object):
    def __init__(self, args):
        self.args = args
        self.events = pd.read_csv("csv/events.csv", parse_dates=["date","start_time","end_time","max_time"])
        self.rios = ["talo", "rank", "pina", "gill", "daws", "fsim", "fsmi", "rabb", "isll", "mcmu"]
        return

    def _check_rio_(self, x):
        bgc = np.median(x.absorption.tolist()[:60])
        peak = np.max(x.absorption)
        c = False
        if peak - bgc > 0.2: c = True
        return c

    def _exact_fdmax_(self, x):
        prcnt = x[x.B_FLUX>5e-6]
        t_fdmax = prcnt.times.tolist()[0]
        return t_fdmax
    
    def to_csv(self, dat, fname="csv/rio_c1.csv"):
        if not os.path.exists(fname): x = pd.DataFrame()
        else: x = pd.read_csv(fname)
        x = x.append(pd.DataFrame.from_records([dat]))
        x = x.drop_duplicates(subset=["pk"], keep="first")
        x.to_csv(fname, header=True, index=False)
        return

    def _case1_(self):
        if self.args.index == -1: 
            os.system("rm -rf data/sim/*")
            os.system("rm csv/rio_c1.csv")
        for i, row in self.events.iterrows():
            if i == self.args.index or self.args.index == -2:
                gfile = "slug/goes/{dn}.csv".format(dn=row["max_time"].strftime("%Y-%m-%d-%H-%M"))
                start, end = row["start_time"]-dt.timedelta(minutes=5), row["end_time"]+dt.timedelta(minutes=5)
                try:
                    _o = GOES.get(row["max_time"], start, end)
                    fmax = np.max(_o.B_FLUX)
                    t_fmax = _o.times.tolist()[_o.B_FLUX.argmax()]
                    t_fdmax = slope_analysis(np.array(_o.B_FLUX), _o.times.tolist(), sec=2, xT=60)    
                    t_fdmax = self._exact_fdmax_(_o)#slope_analysis(np.array(_o.B_FLUX), _o.times.tolist(), sec=2, xT=60)    
                    folder = "data/sim/{dn}/".format(dn=row["max_time"].strftime("%Y-%m-%d-%H-%M"))
                    for rio in self.rios:
                        lat, lon = get_riom_loc(rio)
                        sza = get_sza(lat, lon, row["max_time"])
                        LT = get_LT(lat, lon, row["max_time"])
                        _r = Riometer.get(row["max_time"], rio, start, end)
                        if len(_r) > 0 and self._check_rio_(_r):
                            try:
                                _rm = _r[(_r.times>=t_fdmax) & (_r.times<t_fmax)]
                                t_rdmax = slope_analysis(np.array(_rm.absorption), _rm.times.tolist(), sec=5, xT=30)
                                if not os.path.exists(folder): os.system("mkdir " + folder)
                                plot.plot_rio_ala(_o, _r, t_fdmax, t_rdmax, start, end, folder + rio +".png")
                                if t_rdmax is not None:
                                    print(" Event - %s(%.2f,%.2f), SZA(%.2f), LT(%.2f), %.5f, %s, %.1f"%(rio, lat,
                                    lon, sza, LT, fmax, t_fmax.strftime("%Y-%m-%d-%H-%M"), (t_rdmax - t_fdmax).total_seconds()))
                                    dat = {"lat": np.round(lat,2), "lon": np.round(lon,2), "sza": np.round(sza,2), 
                                            "lt": np.round(LT,2), "fmax": fmax, "event": row["max_time"],
                                            "t_fdmax": t_fdmax, "t_rdmax":t_rdmax, "dt": np.round((t_rdmax - t_fdmax).total_seconds(),2),
                                            "pk": str(i) + "_" + rio}
                                    self.to_csv(dat)
                            except: traceback.print_exc()
                except: pass
        return

    def _delt_(self, x, y):
        def crosscorr(datax, datay, lag=0, wrap=False):
            if wrap:
                shiftedy = datay.shift(lag)
                shiftedy.iloc[:lag] = datay.iloc[-lag:].values
                return datax.corr(shiftedy)
            else: 
                return datax.corr(datay.shift(lag))

        du = np.nan
        n = len(y) if len(y) > len(x) else len(x)
        dx, dy = pd.DataFrame(), pd.DataFrame()
        dx["dat"], dy["dat"] = signal.resample(x, n), signal.resample(y, n)
        seconds = 2
        fps = 90
        u = range(-int(seconds*fps),int(seconds*fps+1))
        rs = [crosscorr(dx.dat, dy.dat, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
        du = -1*u[np.argmax(rs)]*seconds
        return du

    def _case2_(self):
        if self.args.index == -1:
            os.system("rm -rf data/sim/*")
            os.system("rm csv/rio_c2.csv")
        for i, row in self.events.iterrows():
            if i == self.args.index or self.args.index == -2:
                gfile = "slug/goes/{dn}.csv".format(dn=row["max_time"].strftime("%Y-%m-%d-%H-%M"))
                start, end = row["start_time"]-dt.timedelta(minutes=5), row["end_time"]+dt.timedelta(minutes=5)
                try:
                    _o = GOES.get(row["max_time"], start, end)
                    fmax = np.max(_o.B_FLUX)
                    t_fmax = _o.times.tolist()[_o.B_FLUX.argmax()]
                    _o = _o[(_o.times <= t_fmax)]
                    folder = "data/sim/{dn}/".format(dn=row["max_time"].strftime("%Y-%m-%d-%H-%M"))
                    for rio in self.rios:
                        lat, lon = get_riom_loc(rio)
                        sza = get_sza(lat, lon, row["max_time"])
                        LT = get_LT(lat, lon, row["max_time"])
                        _r = Riometer.get(row["max_time"], rio, start, end)
                        if len(_r) > 0 and self._check_rio_(_r):
                            try:
                                _r = _r[(_r.times <= t_fmax)]
                                du = self._delt_(_o.B_FLUX.tolist(), _r.absorption.tolist())
                                if not np.isnan(du):
                                    print(" Event - %s(%.2f,%.2f), SZA(%.2f), LT(%.2f), %.5f, %.1f"%(rio, lat,
                                        lon, sza, LT, fmax, du))
                                    dat = {"lat": np.round(lat,2), "lon": np.round(lon,2), "sza": np.round(sza,2),
                                            "lt": np.round(LT,2), "fmax": fmax, "event": row["max_time"],
                                            "dt": np.round(du,2), "pk": str(i) + "_" + rio}
                                    self.to_csv(dat, fname="csv/rio_c2.csv")
                            except: traceback.print_exc()
                except: traceback.print_exc()
        return

    def _exe_(self):
        if self.args.acase == 1: self._case1_()
        if self.args.acase == 2: self._case2_()
        return

class SDStats(object):
    def __init__(self, args):
        self.args = args
        self.events = pd.read_csv("csv/events.csv", parse_dates=["date","start_time","end_time","max_time"])
        self.rads = ["wal", "bks", "fhe", "fhw", "cve", "cvw", "gbr", "kap", "sas", "pgr"]
        return

    def _exe_(self):
        if self.args.acase == 0: self._case0_()
        if self.args.acase == 1: self._case1_()
        if self.args.acase == 2: self._case2_()
        return

    def _case0_(self):
        for i, row in self.events.iterrows():
            edate, start, end = row["max_time"], row["start_time"], row["end_time"]
            for rad in self.rads:
                gsd._fetch_sd_(edate, rad, start, end, wd=51)
        return

    def _check_rad_(self, x):
        bgc = np.median(x.me.tolist()[:60])
        peak = np.max(x.me)
        c = False
        if peak - bgc > 5: c = True
        return c
    
    def _exact_fdmax_(self, x):
        prcnt = x[x.B_FLUX>5e-6]
        t_fdmax = prcnt.times.tolist()[0]
        return t_fdmax
    
    def to_csv(self, dat, fname="csv/rad_c1.csv"):
        if not os.path.exists(fname): x = pd.DataFrame()
        else: x = pd.read_csv(fname)
        x = x.append(pd.DataFrame.from_records([dat]))
        x = x.drop_duplicates(subset=["pk"], keep="first")
        x.to_csv(fname, header=True, index=False)
        return

    def _case1_(self):
        if self.args.index == -1: 
            os.system("rm -rf data/sim/*")
            os.system("rm csv/rad_c1.csv")
        for i, row in self.events.iterrows():
            if i == self.args.index or self.args.index == -2:
                gfile = "slug/goes/{dn}.csv".format(dn=row["max_time"].strftime("%Y-%m-%d-%H-%M"))
                start, end = row["start_time"]-dt.timedelta(minutes=5), row["end_time"]+dt.timedelta(minutes=5)
                edate = row["max_time"]
                try:
                    _o = GOES.get(row["max_time"], start, end)
                    fmax = np.max(_o.B_FLUX)
                    t_fmax = _o.times.tolist()[_o.B_FLUX.argmax()]
                    t_fdmax = slope_analysis(np.array(_o.B_FLUX), _o.times.tolist(), sec=2, xT=60)    
                    t_fdmax = self._exact_fdmax_(_o)
                    folder = "data/sim/{dn}/".format(dn=row["max_time"].strftime("%Y-%m-%d-%H-%M"))
                    for rad in self.rads:
                        lat, lon = get_radar_loc(rad)
                        sza = get_sza(lat, lon, row["max_time"])
                        LT = get_LT(lat, lon, row["max_time"])
                        _r = gsd._fetch_sd_(edate, rad, start, end)
                        if len(_r) > 0 and self._check_rad_(_r):
                            try:
                                _rm = _r[(_r.time>=t_fdmax) & (_r.time<t_fmax)]
                                t_rdmax = slope_analysis(np.array(_rm.me), _rm.time.tolist(), sec=3, xT=30)
                                if not os.path.exists(folder): os.system("mkdir " + folder)
                                #plot.plot_rio_ala(_o, _r, t_fdmax, t_rdmax, start, end, folder + rio +".png")
                                if t_rdmax is not None:
                                    print(" Event - %s(%.2f,%.2f), SZA(%.2f), LT(%.2f), %.5f, %s, %.1f"%(rad, lat,
                                    lon, sza, LT, fmax, t_fmax.strftime("%Y-%m-%d-%H-%M"), (t_rdmax - t_fdmax).total_seconds()))
                                    dat = {"lat": np.round(lat,2), "lon": np.round(lon,2), "sza": np.round(sza,2),
                                            "lt": np.round(LT,2), "fmax": fmax, "event": row["max_time"],
                                            "t_fdmax": t_fdmax, "t_rdmax":t_rdmax, "dt": np.round((t_rdmax - t_fdmax).total_seconds(),2),
                                            "pk": str(i) + "_" + rad}
                                    self.to_csv(dat)
                            except: traceback.print_exc()
                except: traceback.print_exc()
        return

    def _delt_(self, x, y):
        def crosscorr(datax, datay, lag=0, wrap=False):
            if wrap:
                shiftedy = datay.shift(lag)
                shiftedy.iloc[:lag] = datay.iloc[-lag:].values
                return datax.corr(shiftedy)
            else: 
                return datax.corr(datay.shift(lag))

        du = np.nan
        n = len(y) if len(y) > len(x) else len(x)
        dx, dy = pd.DataFrame(), pd.DataFrame()
        dx["dat"], dy["dat"] = signal.resample(x, n), signal.resample(y, n)
        seconds = 2
        fps = 90
        u = range(-int(seconds*fps),int(seconds*fps+1))
        rs = [crosscorr(dx.dat, dy.dat, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
        du = -1*u[np.argmax(rs)]*seconds
        return du

    def _case2_(self):
        if self.args.index == -1:
            os.system("rm -rf data/sim/*")
            os.system("rm csv/rio_c2.csv")
        for i, row in self.events.iterrows():
            if i == self.args.index or self.args.index == -2:
                gfile = "slug/goes/{dn}.csv".format(dn=row["max_time"].strftime("%Y-%m-%d-%H-%M"))
                start, end = row["start_time"]-dt.timedelta(minutes=5), row["end_time"]+dt.timedelta(minutes=5)
                edate = row["max_time"]
                try:
                    _o = GOES.get(row["max_time"], start, end)
                    fmax = np.max(_o.B_FLUX)
                    t_fmax = _o.times.tolist()[_o.B_FLUX.argmax()]
                    _o = _o[(_o.times <= t_fmax)]
                    folder = "data/sim/{dn}/".format(dn=row["max_time"].strftime("%Y-%m-%d-%H-%M"))
                    for rad in self.rads:
                        lat, lon = get_radar_loc(rad)
                        sza = get_sza(lat, lon, row["max_time"])
                        LT = get_LT(lat, lon, row["max_time"])
                        _r = gsd._fetch_sd_(edate, rad, start, end)
                        if len(_r) > 0 and self._check_rad_(_r):
                            try:
                                _r = _r[(_r.time <= t_fmax)]
                                du = self._delt_(_o.B_FLUX.tolist(), _r.me.tolist())
                                if not np.isnan(du):
                                    print(" Event - %s(%.2f,%.2f), SZA(%.2f), LT(%.2f), %.5f, %.1f"%(rad, lat,
                                        lon, sza, LT, fmax, du))
                                    dat = {"lat": np.round(lat,2), "lon": np.round(lon,2), "sza": np.round(sza,2),
                                            "lt": np.round(LT,2), "fmax": fmax, "event": row["max_time"],
                                            "dt": np.round(du,2), "pk": str(i) + "_" + rad}
                                    self.to_csv(dat, fname="csv/rad_c2.csv")
                            except: traceback.print_exc()
                except: traceback.print_exc()
        return
