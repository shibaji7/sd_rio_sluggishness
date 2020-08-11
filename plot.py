import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacepy.plot as splot
import datetime as dt
import matplotlib.dates as mdates
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy import array

import numpy as np
import analysis as ala
import get_sd_data as gsd

splot.style("spacepy_altgrid")
fontT = {"family": "serif", "color":  "k", "weight": "normal", "size": 8}
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}

from matplotlib import font_manager
ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
matplotlib.rcParams["xtick.color"] = "k"
matplotlib.rcParams["ytick.color"] = "k"
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["mathtext.default"] = "default"


def extrap1d(x,y,kind="linear"):
    """ This method is used to extrapolate 1D paramteres """
    interpolator = interp1d(x,y,kind=kind)
    xs = interpolator.x
    ys = interpolator.y
    def pointwise(x):
        if x < xs[0]: return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]: return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else: return interpolator(x)
    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))
    return ufunclike

def coloring_axes(ax, atype="left", col="red"):
    ax.spines[atype].set_color(col)
    ax.tick_params(axis="y", which="both", colors=col)
    ax.yaxis.label.set_color(col)
    fmt = matplotlib.dates.DateFormatter("%H%M")
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    return ax

def coloring_twaxes(ax, atype="left", col="red", twcol="k"):
    ax.spines[atype].set_color(col)
    ax.tick_params(axis="y", which="both", colors=twcol)
    ax.yaxis.label.set_color(twcol)
    fmt = matplotlib.dates.DateFormatter("%H%M")
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    return ax

def example_riom_plot(ev=dt.datetime(2015,3,11,16,22), stn="ott", 
        start=dt.datetime(2015,3,11,16,10), end=dt.datetime(2015,3,11,16,30)):
    riom = ala.Riometer(ev, stn).fetch()
    gos = ala.GOES(ev).fetch()
    fig, axes = plt.subplots(figsize=(9,3),nrows=1,ncols=3,dpi=120)
    fig.subplots_adjust(wspace=.1)
    
    col = "red"
    ax = coloring_axes(axes[0])
    font["color"] = col
    ax.semilogy(gos.times,gos.b_flux,col,linewidth=0.75)
    ax.axvline(gos.set_index("times").b_flux.idxmax(), color=col, linewidth=0.6)
    ax.set_ylim(1e-6,1e-3)
    ax.set_ylabel("solar flux\n"+r"($wm^{-2}$)",fontdict=font)
    font["color"] = "k"
    ax.set_xlabel("time (ut)",fontdict=font)
    ax = coloring_twaxes(ax.twinx())
    ax.plot(riom.times, riom.absorption,"ko", markersize=1)
    ax.axvline(riom.set_index("times").absorption.idxmax(), color="k", linewidth=0.6)
    ax.grid(false)
    ax.set_xlim(start,end)
    ax.set_ylim(-.1, 3.)
    dx = (riom.set_index("times").absorption.idxmax()-gos.set_index("times").b_flux.idxmax()).total_seconds()
    ax.text(0.54,0.3,r"$\bar{\delta}$=%ds"%(dx),horizontalalignment="center",
                    verticalalignment="center", transform=ax.transaxes,fontdict=fontt, rotation=90)
    ax.set_yticklabels([])
    font["color"] = "darkgreen"
    ax.text(0.7,1.05,"station - ott, 11 march 2015, universal time",horizontalalignment="center",
                            verticalalignment="center", transform=ax.transaxes,fontdict=font)
    font["color"] = "k"
    ax.text(.9,.9,"(a)",horizontalalignment="center",verticalalignment="center", transform=ax.transaxes,fontdict=fontt)

    fslope = ala.slope_analysis(np.log10(gos.B_FLUX),gos.times.tolist())
    rslope = ala.slope_analysis(riom.absorption.tolist(), riom.times.tolist(), xT=120)
    ax = coloring_axes(axes[1])
    ax.semilogy(gos.times,gos.B_FLUX,col,linewidth=0.75)
    ax.axvline(fslope, color=col, linewidth=0.6, ls="--")
    ax.set_ylim(1e-6,1e-3)
    ax.set_yticklabels([])
    ax.set_xlabel("Time (UT)",fontdict=font)
    ax = coloring_twaxes(ax.twinx())
    ax.plot(riom.times, riom.absorption,"ko", markersize=1)
    ax.grid(False)
    ax.set_xlim(start,end)
    ax.set_ylim(-.1, 3.)
    ax.axvline(rslope, color="k", linewidth=0.6, linestyle="--")
    ax.set_yticklabels([])
    dy = (rslope-fslope).total_seconds()
    ax.text(0.27,0.8,r"$\bar{\delta}_s$=%ds"%(dy),horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT, rotation=90)
    ax.text(.9,.9,"(b)",horizontalalignment="center",verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

    dx=40
    ax = coloring_axes(axes[2])
    ax.semilogy(gos.times,gos.B_FLUX,col,linewidth=0.75)
    ax.semilogy(gos.times,np.roll(gos.B_FLUX,dx),"r-.")
    ax.set_ylim(1e-6,1e-3)
    ax.set_yticklabels([])
    ax.set_xlabel("Time (UT)",fontdict=font)
    ax = coloring_twaxes(ax.twinx())
    ax.plot(riom.times, riom.absorption,"ko", markersize=1)
    ax.grid(False)
    ax.set_xlim(start,end)
    ax.set_ylim(-.1, 3.)
    ax.set_ylabel(r"Absorption [$\beta$]" + "\n(in dB)",fontdict=font)
    ax.text(0.2,0.9,r"$\bar{\delta}_c$=%ds"%(2*dx),horizontalalignment="center",
                            verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
    ax.text(0.2,0.83,r"$\rho$=%.2f"%.93,horizontalalignment="center",
                            verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
    ax.text(.9,.9,"(c)",horizontalalignment="center",verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
    
    fig.autofmt_xdate(rotation=30,ha="center")
    fig.savefig("images/example.png",bbox_inches="tight")
    return

def example_rad_plot(ev=dt.datetime(2015,3,11,16,22), stn="bks",
        start=dt.datetime(2015,3,11,16,10), end=dt.datetime(2015,3,11,16,30)):
    sdr = gsd._fetch_sd_(ev, stn, start, end)
    gos = ala.GOES(ev).fetch()
    fig, axes = plt.subplots(figsize=(3,3),nrows=1,ncols=1,dpi=120)
    fmt = matplotlib.dates.DateFormatter("%H%M")

    dx = 25
    fslope = ala.slope_analysis(np.log10(gos.B_FLUX),gos.times.tolist())
    col = "red"
    ax = axes
    ax.spines["left"].set_color(col)
    ax.tick_params(axis="y", which="both", colors=col)
    ax.yaxis.label.set_color(col)
    font["color"] = col
    ax.xaxis.set_major_formatter(fmt)
    ax.semilogy(gos.times,gos.B_FLUX,col,linewidth=0.75)
    ax.semilogy(gos.times,np.roll(gos.B_FLUX,dx),"r-.")
    ax.axvline(fslope, color=col, linewidth=0.6, ls="--")
    ax.set_ylim(1e-6,1e-3)
    ax.set_ylabel("Solar Flux\n"+r"($Wm^{-2}$)",fontdict=font)
    font["color"] = "k"
    ax.set_xlabel("Time (UT)",fontdict=font)

    rslope = ala.slope_analysis(sdr.me,sdr.time.tolist())
    ax = coloring_twaxes(ax.twinx())
    ax.plot(sdr.time, sdr.me, "k-")
    ax.set_xlim(start,end)
    ax.axvline(rslope, color="k", linewidth=0.6, linestyle="--")
    ax.set_ylabel(r"Inverse #-GS",fontdict=font)
    ax.grid(False)
    ax.set_xlim(start,end)
    ax.set_ylim(-.1, 20.)

    dy = (rslope-fslope).total_seconds()
    ax.text(0.38,0.8,r"$\bar{\delta}_s$=%ds"%(dy),horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT, rotation=90)
    ax.text(0.2,0.9,r"$\bar{\delta}_c$=%ds"%(2*dx),horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
    ax.text(0.2,0.83,r"$\rho$=%.2f"%.56,horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
    ax.text(0.78,0.9,"Station - BKS",horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

    fig.autofmt_xdate(rotation=30,ha="center")
    fig.savefig("images/example_sd.png",bbox_inches="tight")
    return

def example_hrx_plot(ev=dt.datetime(2015,3,11,16,22), stn="ott",
        start=dt.datetime(2015,3,11,16,10), end=dt.datetime(2015,3,11,16,30)):
    riom = ala.Riometer(ev, stn).fetch()
    gos = ala.GOES(ev).fetch()
    fig, axes = plt.subplots(figsize=(3,6),nrows=2,ncols=1,dpi=120)
    fig.subplots_adjust(hspace=0.1)
    fmt = matplotlib.dates.DateFormatter("%H%M")

    fslope = ala.slope_analysis(np.log10(gos.B_FLUX),gos.times.tolist())
    col = "red"
    ax = coloring_axes(axes[0])
    font["color"] = col
    ax.semilogy(gos.times,gos.B_FLUX,col,linewidth=0.75)
    ax.axvline(gos.set_index("times").B_FLUX.idxmax(), color=col, linewidth=0.6)
    ax.axvline(fslope, color=col, linewidth=0.6, ls="--")
    ax.set_ylim(1e-8,1e-3)
    ax.set_ylabel("Soft X-ray [0.1-0.8 nm]\n"+r"($Wm^{-2}$)",fontdict=font)
    font["color"] = "k"
    ax.set_xlabel("Time (UT)",fontdict=font)
    
    ax = coloring_twaxes(ax.twinx())
    rslope = ala.slope_analysis(riom.absorption.tolist(), riom.times.tolist(), xT=120)
    ax.plot(riom.times, riom.absorption,"ko", markersize=1)
    ax.axvline(riom.set_index("times").absorption.idxmax(), color="k", linewidth=0.6)
    ax.grid(False)
    ax.set_xlim(start,end)
    ax.set_ylim(-.1, 3.)
    ax.axvline(rslope, color="k", linewidth=0.6, linestyle="--")
    ax.set_ylabel(r"Absorption [$\beta$]" + "\n(in dB)",fontdict=font)
    ax.set_ylim(-.1, 3.)
    dx = (riom.set_index("times").absorption.idxmax()-gos.set_index("times").B_FLUX.idxmax()).total_seconds()
    dy = (rslope-fslope).total_seconds()
    ax.text(0.36,0.85,r"$\bar{\delta}_s$=%ds"%(dy),horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT, rotation=90)
    ax.text(0.68,0.25,r"$\bar{\delta}$=%ds"%(dx),horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT, rotation=90)
    ax.text(0.8,0.9,"Station - OTT",horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
   
    fslope = ala.slope_analysis(np.log10(gos.A_FLUX),gos.times.tolist())
    col = "blue"
    ax = coloring_axes(axes[1])
    font["color"] = col
    ax.semilogy(gos.times,gos.A_FLUX,col,linewidth=0.75)
    ax.axvline(gos.set_index("times").A_FLUX.idxmax(), color=col, linewidth=0.6)
    ax.axvline(fslope, color=col, linewidth=0.6, ls="--")
    ax.set_ylim(1e-8,1e-3)
    ax.set_ylabel("Hard X-ray [0.05-0.4 nm]\n"+r"($Wm^{-2}$)",fontdict=font)
    font["color"] = "k"
    ax.set_xlabel("Time (UT)",fontdict=font)
    
    ax = coloring_twaxes(ax.twinx())
    rslope = ala.slope_analysis(riom.absorption.tolist(), riom.times.tolist(), xT=120)
    ax.plot(riom.times, riom.absorption,"ko", markersize=1)
    ax.axvline(riom.set_index("times").absorption.idxmax(), color="k", linewidth=0.6)
    ax.grid(False)
    ax.set_xlim(start,end)
    ax.set_ylim(-.1, 3.)
    ax.axvline(rslope, color="k", linewidth=0.6, linestyle="--")
    ax.set_ylabel(r"Absorption [$\beta$]" + "\n(in dB)",fontdict=font)
    ax.set_ylim(-.1, 3.)
    dx = (riom.set_index("times").absorption.idxmax()-gos.set_index("times").B_FLUX.idxmax()).total_seconds()
    dy = (rslope-fslope).total_seconds()
    ax.text(0.36,0.85,r"$\bar{\delta}_s$=%ds"%(dy),horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT, rotation=90)
    ax.text(0.68,0.25,r"$\bar{\delta}$=%ds"%(dx),horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT, rotation=90)
    ax.text(0.8,0.9,"Station - OTT",horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
    ax.text(0.1,0.9,"(b)",horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

    ax = axes[0]
    ax.axvline(fslope, color="b", linewidth=0.6, linestyle="--",clip_on=False,ymin=-.2)
    ax.axvline(gos.set_index("times").A_FLUX.idxmax(), color="b", linewidth=0.6,clip_on=False,ymin=-.2)
    ax.text(0.1,0.9,"(a)",horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
    fig.autofmt_xdate(rotation=30,ha="center")
    fig.savefig("images/example_hrx.png",bbox_inches="tight")
    return

class Statistics(object):
    def __init__(self, args):
        if args.acase == 0 and not args.rad:
            fname = "csv/rio_c0.csv"
            self.fname = "images/stat_rio_c0.png"
        if args.acase == 1 and not args.rad:
            fname = "csv/rio_c1.csv"
            self.fname = "images/stat_rio_c1.png"
        if args.acase == 2 and not args.rad:
            fname = "csv/rio_c2.csv"
            self.fname = "images/stat_rio_c2.png"
        if args.acase == 1 and args.rad:
            fname = "csv/rad_c1.csv"
            self.fname = "images/stat_rad_c1.png"
        if args.acase == 2 and args.rad:
            fname = "csv/rad_c2.csv"
            self.fname = "images/stat_rad_c2.png"
        self.dat = pd.read_csv(fname)
        self.args = args
        return

    def _model_(self, X, y, family=sm.families.NegativeBinomial()):
        model = sm.GLM(y, X, family=family)
        response = model.fit()
        return response

    def _create_x_(self, cossza, lat, logfmax, lt, lp="cossza"):
        _o = pd.DataFrame()
        if lp == "cossza": 
            L = len(cossza)
            _o["cossza"], _o["lat"], _o["lt"], _o["logfmax"] = cossza, [lat]*L, [lt]*L, [logfmax]*L
        if lp == "lat":
            L = len(lat)
            _o["cossza"], _o["lat"], _o["lt"], _o["logfmax"] = [cossza]*L, lat, [lt]*L, [logfmax]*L
        if lp == "logfmax":
            L = len(logfmax)
            _o["cossza"], _o["lat"], _o["lt"], _o["logfmax"] = [cossza]*L, [lat]*L, [lt]*L, logfmax
        if lp == "lt":
            L = len(lt)
            _o["cossza"], _o["lat"], _o["lt"], _o["logfmax"] = [cossza]*L, [lat]*L, lt, [logfmax]*L
        return _o

    def _img_(self):
        fig, axes = plt.subplots(figsize=(5,5.5),nrows=2,ncols=2,dpi=130, sharey="row")
        fig.subplots_adjust(hspace=0.3)
        self.dat.dt = np.abs(self.dat.dt)
        df = self.dat
        if self.args.acase == 1 and not self.args.rad:
            df = df[(df.dt>20) & (df.sza<140) & (df.sza>60)]
            ymax, ymin = 130, 50
        if self.args.acase == 2 and not self.args.rad:
            df = df[(df.dt>200) & (df.sza<140) & (df.sza>60)]
        if self.args.acase == 2 and self.args.rad:
            df = df[(df.dt!=0) & (df.dt!=360)]
            szamax, szamin = 130, 50
        df["cossza"] = df.sza.transform(lambda x: np.cos(np.deg2rad(x)))
        df["logfmax"] = df.fmax.transform(lambda x: np.log10(x))
        df = df.sort_values(by="sza")
        r = self._model_(df[["cossza","lat","logfmax","lt"]].values, df.dt)
        x = self._create_x_(df.cossza.tolist(), np.mean(df.lat), np.mean(df.logfmax), np.mean(df["lt"]), lp="cossza")
        ax = axes[0, 0]
        ax.semilogy(df.sza, df.dt, linestyle="None", marker="o", markersize=2, color="k", linewidth=1, alpha=0.5)
        o = r.get_prediction(x[["cossza","lat","logfmax","lt"]].values)
        m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
        ax.plot(df.sza, o.predicted_mean, "r-", linewidth=0.75, alpha=0.8)
        ax.fill_between(df.sza, m - 1.98*v, m + 1.98*v, color="r", alpha=0.2)
        ax.set_xlim(50,130)
        ax.set_ylabel(r"$\bar{\delta}_s$ (sec)",fontdict=font)
        ax.set_xlabel(r"$\chi$"+" (deg)",fontdict=font)
        ax.text(0.9,0.9,"(a)", horizontalalignment="center",
                verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

        df = df.sort_values(by="lat")
        r = self._model_(df[["cossza","lat","logfmax","lt"]].values, df.dt)
        x = self._create_x_(np.mean(df.cossza), df.lat.tolist(), np.mean(df.logfmax), np.mean(df["lt"]), lp="lat")
        ax = axes[0, 1]
        ax.semilogy(df.lat, df.dt, linestyle="None", marker="o", markersize=2, color="k", linewidth=1, alpha=0.5)
        o = r.get_prediction(x[["cossza","lat","logfmax","lt"]].values)
        m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
        ax.plot(df.lat, o.predicted_mean, "r-", linewidth=0.75, alpha=0.8)
        ax.fill_between(df.lat, m - 1.98*v, m + 1.98*v, color="r", alpha=0.2)
        ax.set_xlabel(r"$\phi$"+" (deg)",fontdict=font)
        ax.text(0.9,0.9,"(b)", horizontalalignment="center",
                verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

        df = df.sort_values(by="lt")
        r = self._model_(df[["cossza","lat","logfmax","lt"]].values, df.dt)
        x = self._create_x_(np.mean(df.cossza), np.mean(df.lat), np.mean(df.logfmax), df["lt"].tolist(), lp="lt")
        ax = axes[1,0]
        ax.semilogy(df["lt"], df.dt, linestyle="None", marker="o", markersize=2, color="k", linewidth=1, alpha=0.5)
        o = r.get_prediction(x[["cossza","lat","logfmax","lt"]].values)
        m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
        ax.plot(df["lt"], o.predicted_mean, "r-", linewidth=0.75, alpha=0.8)
        ax.fill_between(df["lt"], m - 1.98*v, m + 1.98*v, color="r", alpha=0.2)
        ax.set_xlim(0, 24)
        ax.set_ylabel(r"$\bar{\delta}_s$ (sec)",fontdict=font)
        ax.set_xlabel("LT (Hours)",fontdict=font)
        ax.text(0.9,0.9,"(c)", horizontalalignment="center",
                verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

        df = df.sort_values(by="logfmax")
        r = self._model_(df[["cossza","lat","logfmax","lt"]].values, df.dt)
        x = self._create_x_(np.mean(df.cossza), np.mean(df.lat), df.logfmax.tolist(), np.mean(df["lt"]), lp="logfmax")
        ax = axes[1,1]
        ax.loglog(df.fmax, df.dt, linestyle="None", marker="o", markersize=2, color="k", linewidth=1, alpha=0.5)
        o = r.get_prediction(x[["cossza","lat","logfmax","lt"]].values)
        m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
        ax.plot(df.fmax, o.predicted_mean, "r-", linewidth=0.75, alpha=0.8)
        ax.fill_between(df.fmax, m - 1.98*v, m + 1.98*v, color="r", alpha=0.2)
        ax.set_xlim(1e-4,1e-3)
        ax.tick_params(axis="x", which="both", rotation=30)
        ax.set_xlabel(r"$I_{\infty}^{max}$"+" ("+r"$Wm^{-2}$)",fontdict=font)
        ax.text(0.9,0.9,"(d)", horizontalalignment="center",
                verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

        fig.savefig(self.fname, bbox_inches="tight")
        ala.anova_ml(self.args)
        return

    def _fig_(self):
        fig, axes = plt.subplots(figsize=(10,6),nrows=3,ncols=5,dpi=150, sharey="row", sharex="col")
        tags = [["(a-1)","(b-1)","(c-1)","(d-1)","(e-1)"],
                ["(a-2)","(b-2)","(c-2)","(d-2)","(e-2)"],
                ["(a-3)","(b-3)","(c-3)","(d-3)","(e-3)"],]
        dat = self.dat
        for i, llim, ulim in zip(range(3), [1e-7, 1e-5, 1e-4], [1e-5, 1e-4, 1e-2]):
            df = dat[(dat.fmax > llim) & (dat.fmax < ulim)]
            df["cossza"] = df.sza.transform(lambda x: np.cos(np.deg2rad(x)))
            df["logfmax"] = df.fmax.transform(lambda x: np.log10(x))
            df = df.sort_values(by="sza")
            r = self._model_(df[["cossza","lat","logfmax","lt"]].values, df.dt)
            x = self._create_x_(df.cossza.tolist(), np.mean(df.lat), np.mean(df.logfmax), np.mean(df["lt"]), lp="cossza")
            ax = axes[i,0]
            ax.semilogy(df.sza, df.dt, linestyle="None", marker="o", markersize=2, color="k", linewidth=1, alpha=0.5)
            o = r.get_prediction(x[["cossza","lat","logfmax","lt"]].values)
            m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
            ax.plot(df.sza, o.predicted_mean, "r-", linewidth=0.75, alpha=0.8)
            ax.fill_between(df.sza, m - 1.98*v, m + 1.98*v, color="r", alpha=0.2)
            ax.set_xlim(20,80)
            ax.set_ylim(1e2,1e4)
            ax.text(0.9,0.9,tags[i][0], horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

            df = df.sort_values(by="lat")
            r = self._model_(df[["cossza","lat","logfmax","lt"]].values, df.dt)
            x = self._create_x_(np.mean(df.cossza), df.lat.tolist(), np.mean(df.logfmax), np.mean(df["lt"]), lp="lat")
            ax = axes[i,1]
            ax.semilogy(df.lat, df.dt, linestyle="None", marker="o", markersize=2, color="k", linewidth=1, alpha=0.5)
            o = r.get_prediction(x[["cossza","lat","logfmax","lt"]].values)
            m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
            ax.plot(df.lat, o.predicted_mean, "r-", linewidth=0.75, alpha=0.8)
            ax.fill_between(df.lat, m - 1.98*v, m + 1.98*v, color="r", alpha=0.2)
            ax.set_ylim(1e2,1e4)
            ax.text(0.9,0.9,tags[i][1], horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
            
            df = df.sort_values(by="lt")
            r = self._model_(df[["cossza","lat","logfmax","lt"]].values, df.dt)
            x = self._create_x_(np.mean(df.cossza), np.mean(df.lat), np.mean(df.logfmax), df["lt"].tolist(), lp="lt")
            ax = axes[i,2]
            ax.semilogy(df["lt"], df.dt, linestyle="None", marker="o", markersize=2, color="k", linewidth=1, alpha=0.5)
            o = r.get_prediction(x[["cossza","lat","logfmax","lt"]].values)
            m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
            ax.plot(df["lt"], o.predicted_mean, "r-", linewidth=0.75, alpha=0.8)
            ax.fill_between(df["lt"], m - 1.98*v, m + 1.98*v, color="r", alpha=0.2)
            ax.set_ylim(1e2,1e4)
            ax.set_xlim(0, 24)
            ax.text(0.9,0.9,tags[i][2], horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

            df = df.sort_values(by="logfmax")
            r = self._model_(df[["cossza","lat","logfmax","lt"]].values, df.dt)
            x = self._create_x_(np.mean(df.cossza), np.mean(df.lat), df.logfmax.tolist(), np.mean(df["lt"]), lp="logfmax")
            ax = axes[i,3]
            ax.loglog(df.fmax, df.dt, linestyle="None", marker="o", markersize=2, color="k", linewidth=1, alpha=0.5)
            o = r.get_prediction(x[["cossza","lat","logfmax","lt"]].values)
            m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
            ax.plot(df.fmax, o.predicted_mean, "r-", linewidth=0.75, alpha=0.8)
            ax.fill_between(df.fmax, m - 1.98*v, m + 1.98*v, color="r", alpha=0.2)
            ax.set_ylim(1e2,1e4)
            ax.set_xlim(1e-6,1e-3)
            ax.text(0.9,0.9,tags[i][3], horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

            ax = axes[i,4]
            o = r.get_prediction(df[["cossza","lat","logfmax","lt"]].values)
            m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
            ax.loglog(df.dt, m, "ko", markersize=0.8, alpha=0.6)
            ax.loglog([100,20000], [100,20000], "r", linewidth=0.6)
            ax.set_ylim(1e2,1e4)
            ax.set_xlim(1e2,1e4)
            ax.text(0.9,0.9,tags[i][4], horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
            ax.text(0.2,0.9,r"$\rho$=%.2f"%np.corrcoef(m, df.dt)[0,1],horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

        axes[0,0].set_ylabel(r"$\bar{\delta}$ (sec)",fontdict=font)
        axes[1,0].set_ylabel(r"$\bar{\delta}$ (sec)",fontdict=font)
        axes[2,0].set_ylabel(r"$\bar{\delta}$ (sec)",fontdict=font)

        axes[2,0].set_xlabel(r"$\chi$"+" (deg)",fontdict=font)
        axes[2,1].set_xlabel(r"$\phi$"+" (deg)",fontdict=font)
        axes[2,2].set_xlabel("LT (Hours)",fontdict=font)
        axes[2,3].set_xlabel(r"$I_{\infty}^{max}$"+" ("+r"$Wm^{-2}$)",fontdict=font)
        axes[2,4].set_xlabel(r"$\hat{\bar{\delta}}$ (sec)",fontdict=font)

        axes[0,4].text(1.05,0.5,"Flare Class=C",horizontalalignment="center",
                            verticalalignment="center", transform=axes[0,4].transAxes,fontdict=fontT,rotation=90)
        axes[1,4].text(1.05,0.5,"Flare Class=M",horizontalalignment="center",
                            verticalalignment="center", transform=axes[1,4].transAxes,fontdict=fontT,rotation=90)
        axes[2,4].text(1.05,0.5,"Flare Class=X",horizontalalignment="center",
                            verticalalignment="center", transform=axes[2,4].transAxes,fontdict=fontT,rotation=90)

        fig.subplots_adjust(hspace=0.1,wspace=.1)
        fig.savefig(self.fname, bbox_inches="tight")
        ala.anova(self.args)
        return

def plot_rio_ala(gos, riom, fslope, rslope, start, end,  fname):
    fig, ax = plt.subplots(figsize=(3,3),nrows=1,ncols=1,dpi=120)
    col="red"
    ax = coloring_axes(ax)
    ax.semilogy(gos.times,gos.B_FLUX,col,linewidth=0.75)
    ax.axvline(fslope, color=col, linewidth=0.6, ls="--")
    ax.set_ylim(1e-6,1e-3)
    ax.set_xlabel("Time (UT)",fontdict=font)
    ax.set_ylabel(r"$Wm^{-2}$",fontdict=font)
    ax = coloring_twaxes(ax.twinx())
    ax.plot(riom.times, riom.absorption,"ko", markersize=1)
    ax.grid(False)
    ax.set_xlim(start,end)
    ax.set_ylim(-.1, 3.)
    if rslope is not None:
        ax.axvline(rslope, color="k", linewidth=0.6, linestyle="--")
        ax.set_ylabel(r"$\beta$, dB",fontdict=font)
        dy = (rslope-fslope).total_seconds()
        ax.text(0.27,0.8,r"$\bar{\delta}_s$=%ds"%(dy),horizontalalignment="center",
                verticalalignment="center", transform=ax.transAxes,fontdict=fontT, rotation=90)
    fig.savefig(fname, bbox_inches="tight")
    return

def plot_cdf():
    from scipy.stats import kstest
    import scipy.stats as stats
    files = ["csv/rio_c0.csv", "csv/rio_c1.csv", "csv/rio_c2.csv", "csv/rad_c1.csv", "csv/rad_c2.csv"]
    labels = [r"$\bar{\delta}^{rio}$", r"$\bar{\delta}_{s}^{rio}$", r"$\bar{\delta}_{c}^{rio}$",
            r"$\bar{\delta}_{s}^{rad}$", r"$\bar{\delta}_{c}^{rad}$"]
    fig, ax = plt.subplots(figsize=(3,3),nrows=1,ncols=1,dpi=90)
    C = ["r", "darkgreen", "blue", "black", "m"]
    for l, f in enumerate(files):
        dat = pd.read_csv(f)
        dat = dat[np.abs(dat.dt)>0]
        x = np.log10(np.abs(dat.dt))
        x, y = sorted(x), np.arange(len(x)) / len(x)
        ax.plot(x, y, lw=0.75, alpha=0.7, label=labels[l], color=C[l])
        ax.set_xlim(1,4)
        ax.set_ylim(0,1)
        for fu in files:
            du = pd.read_csv(fu)
            xu = np.log10(np.abs(du.dt))
            xu = sorted(xu)
            print(kstest(x, xu))
    ax.legend(loc=4, fontsize=7)
    ax.set_xlabel(r"$log_{10}(\bar{\delta}_{*})$ (sec)",fontdict=font)
    ax.set_ylabel(r"CDF",fontdict=font)

    #for l, f in enumerate(files):
    #    dat = pd.read_csv(f)
    #    dat = dat[np.abs(dat.dt)>0]
    #    x = np.log10(np.abs(dat.dt))
    #    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(x)
    #    mx = np.linspace(1,4,100)
    #    ax.plot(mx, stats.gamma.pdf(mx, fit_alpha, fit_loc, fit_beta), lw=0.75, alpha=0.7, label=labels[l])
    #ax.legend(loc=4, fontsize=8)
    fig.savefig("images/cdf.png", bbox_inches="tight")
    return
