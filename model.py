import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import spacepy.plot as splot
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import array
from scipy.stats import linregress

import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd

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

c = 3e8 #m/s
w = 2*np.pi*30e6 #rad/s
me = 9.31e-31 #kg
q = 1.60218e-19 #C
kB = 1.38e-23 #JK
g = 9.8 #m/s^2

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

def eformat(f, prec, exp_digits):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split("e")
    u = mantissa + r"$\times 10^{%d}$"%(int(exp))
    return u

def func_aeff(iq, ne, d, T=275, theta=70.):
    m = 4.8e-26
    rho = 34 * q
    aeff = 0.375 / ( d * ( ne - (iq*np.cos(np.deg2rad(theta))*g*m*d / (rho*np.e*kB*T)) ) )
    print(aeff)
    return aeff

def fetch_electron_densities(Iq):
    def get_params(Iq):
        b, h = 0., 0.
        xn = 10*np.log10(Iq/1e-6)
        d = pd.read_csv("csv/beta.csv")
        x, y = d["phi"], d["beta"]
        b = extrap1d(x, y)(xn)
        d = pd.read_csv("csv/height.csv")
        x, y = d["phi"], d["height"]
        h = extrap1d(x, y)(xn)
        return b, h
    df = pd.read_csv("csv/rio_c0.csv")
    df = df[(df.dt>100) & (df.dt<800)].dropna()
    df["cossza"] = df.sza.transform(lambda x: np.cos(np.deg2rad(x)))
    df["logfmax"] = df.fmax.transform(lambda x: np.log10(x))
    model = sm.GLM(df.dt, df[["logfmax"]].values, family=sm.families.NegativeBinomial())
    r = model.fit()
    o = r.get_prediction(np.log10(Iq))
    m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
    ne = 0.
    b, h = get_params(Iq)
    ne = np.mean(np.array([1.43e13*np.exp(-0.15*h + ((b-0.15) * (z-h))) for z in np.arange(70,75)]), axis=0)
    return ne, m, v

Iq = 10**np.arange(-6,-2,0.01)
ne, dtm, dtv = fetch_electron_densities(Iq)
fig, axes = plt.subplots(figsize=(2,4),dpi=130, nrows=2, ncols=1)
fig.subplots_adjust(hspace=.1)
ax = axes[0]
col = "red"
ax.spines["left"].set_color(col)
ax.tick_params(axis="y", which="both", colors=col)
ax.yaxis.label.set_color(col)
font["color"] = col
ax.loglog(Iq, ne, "r--")
ax.set_ylim(1e9,1e13)
ax.set_xlim(1e-6,1e-2)
ax.set_ylabel(r"$N_e^{max}$ (in "+r"$m^{-3}$)",fontdict=font)
ax.text(0.9,0.5,"(a)",horizontalalignment="center",
        verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
ax = ax.twinx()
ax.spines["left"].set_color(col)
col = "blue"
ax.spines["right"].set_color(col)
ax.tick_params(axis="y", which="both", colors=col)
ax.yaxis.label.set_color(col)
font["color"] = col
ax.loglog(Iq, dtm, "blue",linewidth=0.75)
ax.grid(False)
ax.set_xlim(1e-6,1e-2)
ax.set_ylabel(r"$\bar{\delta}^{rio}_{fitted}$ (in "+r"sec)",fontdict=font)
ax.set_xticklabels([])

ax = axes[1]
font["color"] = "k"
ax.set_xlabel(r"$I_{\infty}^{max}$"+" (in "+r"$Wm^{-2}$)",fontdict=font)
ax.set_ylabel(r"$\alpha_{eff}$ (in "+r"$m^{3} sec^{-1})$",fontdict=font)
ax.axhspan(ymin=1e-13,ymax=3e-13,color="r",alpha=0.6)
ax.axhspan(ymax=1e-11,ymin=1e-12,color="b",alpha=0.6)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylim(1e-16,1e-10)
ax.set_xlim(1e-6,1e-2)
ax.axvline(1e-5,color="orange",ls="--",linewidth=0.75)
ax.axvline(1e-4,color="red",ls="--",linewidth=0.75)
fontT["color"] = "darkgreen"
ax.text(3e-6,3e-11,"C", horizontalalignment="center", verticalalignment="center", fontdict=fontT)
fontT["color"] = "orange"
ax.text(3e-5,3e-11,"M", horizontalalignment="center", verticalalignment="center", fontdict=fontT)
fontT["color"] = "red"
ax.text(3e-4,3e-11,"X", horizontalalignment="center", verticalalignment="center", fontdict=fontT)
fontT["size"]=8
ae = func_aeff(Iq, ne, dtm, T=370, theta=80)
ax.loglog(Iq, ae,"k", linewidth=0.75)
fontT["color"] = "k"
fontT["size"]=6
slope, intercept, r_value, p_value, std_err = linregress(np.log10(Iq), np.log10(ae))
ax.text(1.05,0.5,r"m=-%s $m^3s^{-1}/10$ $Wm^{-2}$"%eformat(10**slope, 2, 2), horizontalalignment="center",
        verticalalignment="center", transform=ax.transAxes,fontdict=fontT,rotation=90)
fontT["size"]=8
ax.text(0.9,0.5,"(b)",horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,fontdict=fontT)

fig.savefig("images/simulation.png",bbox_inches="tight")
os.system("rm -rf __pycache__/")
