import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import spacepy.plot as splot
import matplotlib.pyplot as plt

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

def eformat(f, prec, exp_digits):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split("e")
    u = mantissa + r"$\times 10^{%d}$"%(int(exp))
    return u

def func_aeff(iq, ne, d, T=275, theta=70.):
    g = 9.8
    m = 4.8e-26
    rho = 34 * 1.60218e-19
    e = np.e
    k = 1.38e-23
    cos = np.cos(np.deg2rad(theta))
    aeff = 0.375 / ( d * ( ne - (iq*cos*g*m*d / (rho*e*k*T)) ) )
    return aeff

def fetch_electron_densities(Iq):
    df = pd.read_csv("csv/rio_c0.csv")
    df["cossza"] = df.sza.transform(lambda x: np.cos(np.deg2rad(x)))
    df["logfmax"] = df.fmax.transform(lambda x: np.log10(x))
    model = sm.GLM(df.dt, df[["logfmax"]].values, family=sm.families.NegativeBinomial())
    r = model.fit()
    o = r.get_prediction(np.log10(Iq))
    m, v = o.predicted_mean, np.sqrt(o.var_pred_mean)
    ne = 0.
    return ne, m

Iq = 10**np.arange(-6,-2,0.1)
fig, axes = plt.subplots(figsize=(2,4),dpi=130, nrows=2, ncols=1)
fig.subplots_adjust(hspace=.1)
ax = axes[0]
col = "red"
ax.spines["left"].set_color(col)
ax.tick_params(axis='y', which='both', colors=col)
ax.yaxis.label.set_color(col)
font["color"] = col
#ax.loglog(Iq, fetch_electron_densities(Iq,ssn=140)/20, "r--")
ax.set_ylim(1e7,1e11)
ax.set_xlim(1e-6,1e-3)
ax.set_ylabel(r"$N_e^{max}$ (in "+r"$m^{-3}$)",fontdict=font)
ax.text(0.9,0.5,"(a)",horizontalalignment="center",
                verticalalignment="center", transform=ax.transAxes,fontdict=fontT)
ax = ax.twinx()
ax.spines["left"].set_color(col)
col = "blue"
ax.spines['right'].set_color(col)
ax.tick_params(axis="y", which="both", colors=col)
ax.yaxis.label.set_color(col)
font["color"] = col
ax.loglog(Iq, fetch_electron_densities(Iq)[1], "blue",linewidth=0.75)
ax.grid(False)
ax.set_xlim(1e-6,1e-2)
ax.set_ylabel(r"$\bar{\delta}$ (in "+r"sec)",fontdict=font)
ax.set_xticklabels([])

ax = axes[1]

fig.savefig("images/simulation.png",bbox_inches="tight")
os.system("rm -rf __pycache__/")
