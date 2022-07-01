'''
This script makes the SDSS spectra overlapped on SPLUS photometry
'''
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord 
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sn
import glob
import argparse
import sys
import os
from astropy.visualization import hist
from astroML.datasets import fetch_imaging_sample, fetch_sdss_S82standards
from astroML.crossmatch import crossmatch_angular
sn.set_context("poster")


# Read the files
parser = argparse.ArgumentParser(
    description="""Make a spectras""")

parser.add_argument("fileSdss", type=str,
                    default="teste-program",
                    help="Name of file, taken the prefix")

parser.add_argument("TableSplus", type=str,
                    default="teste-program",
                    help="Name of table, taken the prefix")

parser.add_argument("--Object", type=str,
                    default=None,
                    help="Id object of the source under interest ")

parser.add_argument("--ymin", required=False, type=float, default=None,
                    help="""Value y-axis min""")

parser.add_argument("--ymax", required=False, type=float, default=None,
                    help="""Value y-axis max""")

cmd_args = parser.parse_args()
file_spec = cmd_args.fileSdss + ".fits"
file_table = cmd_args.TableSplus + ".ecsv"

# SDSS spectra
datadir_sdss = "sdss-spectra/"
try:
    hdu = fits.open(file_spec)
except FileNotFoundError:
    hdu = fits.open(os.path.join(datadir_sdss, file_spec))

# Table
datadir = "../"

try:
    table = Table.read(os.path.join(datadir, file_table), format="ascii.ecsv")
except FileNotFoundError:
    file_table = cmd_args.TableSplus + ".csv"
    df = pd.read_csv(os.path.join(datadir, file_table))
    # Converting pandas into astropy
    table = Table.from_pandas(df)


# If I want the spectra of an particular objetc
idd1 = []
for i in table:
    id1 = i["ID"]
    idd1.append(id1)

if cmd_args.Object is not None:
    Object_ = cmd_args.Object
    mask = np.array([source in Object_ for source in idd1])
    table = table[mask]
else:
    table = table

# Data from the SDSS spectra
hdudata = hdu[1].data
wl = 10**hdudata.field("loglam")
Flux = 1E-17*hdudata.field("flux")

# Data of the SPLUs list
mag_br, mag_err_br, mag_nr, mag_err_nr = [], [], [], []
#wl_sp = [3485, 3785, 3950, 4100, 4300, 4803, 5150, 6250, 6600, 7660, 8610, 9110]
#color = ["#CC00FF", "#9900FF", "#6600FF", "#0000FF", "#009999", "#006600", "#DD8000", "#FF0000", "#CC0066", "#990033", "#660033", "#330034"]
#marker = ["s", "o", "o", "o", "o", "s", "o", "s", "o", "s", "o", "s"] ### tienen todos los filtros

wl_br = [3485, 4803,  6250,  7660,  9110]
wl_nr = [3785, 3950, 4100, 4300, 5150,  6600, 8610]
color_br = ["#CC00FF", "#006600",  "#FF0000", "#990033",  "#330034"]
color_nr = ["#9900FF", "#6600FF", "#0000FF", "#009999",  "#DD8000",  "#CC0066", "#660033"]
marker_br = ["s", "s",  "s", "s", "s"]
marker_nr = ["o", "o", "o", "o", "o",  "o", "o"]

mag_br.append(table["u_PStotal"]) 
mag_nr.append(table["J0378_PStotal"])
mag_nr.append(table["J0395_PStotal"])
mag_nr.append(table["J0410_PStotal"])
mag_nr.append(table["J0430_PStotal"])
mag_br.append(table["g_PStotal"])
mag_nr.append(table["J0515_PStotal"]) 
mag_br.append(table["r_PStotal"]) 
mag_nr.append(table["J0660_PStotal"])
mag_br.append(table["i_PStotal"]) 
mag_nr.append(table["J0861_PStotal"]) 
mag_br.append(table["z_PStotal"])

#ERRO PStotal
mag_err_br.append(table["e_u_PStotal"])
mag_err_nr.append(table["e_J0378_PStotal"])
mag_err_nr.append(table["e_J0395_PStotal"])
mag_err_nr.append(table["e_J0410_PStotal"])
mag_err_nr.append(table["e_J0430_PStotal"])
mag_err_br.append(table["e_g_PStotal"])
mag_err_nr.append(table["e_J0515_PStotal"])
mag_err_br.append(table["e_r_PStotal"])
mag_err_nr.append(table["e_J0660_PStotal"]) 
mag_err_br.append(table["e_i_PStotal"])
mag_err_nr.append(table["e_J0861_PStotal"])
mag_err_br.append(table["e_z_PStotal"])

# ff = (10**(-(table["R_PStotal"][ind] + 2.41) / 2.5)) / 6250.0**2
# print(ff)
# for i, ii in zip(wl, Flux):
#     if i> 6000 and i< 6300:
#          print(i, ii)

# Find scale factor
m = wl == 6250.289 
wl_part = wl[m]
flux_part = Flux[m]
Fsp = (10**(-(table["r_PStotal"] + 2.41) / 2.5)) / 6250.0**2
factor = flux_part / Fsp

# Propagation of error
err_br = []
for wll, magg, magerr in zip(wl_br, mag_br, mag_err_br):
    c = (10**(-2.41/2.5)) / wll**2
    c /= 1e-15
    b = -(1. / 2.5)
    err = np.sqrt(((c*10**(b*magg))**2)*(np.log(10)*b*magerr)**2)
    err_br.append(err)

err_nr = []
for wll_nr, magg_nr, magerr_nr in zip(wl_nr, mag_nr, mag_err_nr):
    c_nr = (10**(-2.41/2.5)) / wll_nr**2
    c_nr /= 1e-15
    b_nr = -(1. / 2.5)
    err_nr_ = np.sqrt(((c_nr*10**(b_nr*magg_nr))**2)*(np.log(10)*b_nr*magerr_nr)**2)
    err_nr.append(err_nr_)

# PLOTS
fig, ax = plt.subplots(figsize=(12, 9))
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
ax.set(xlim=[3000,9700])

#axis limit
mask_lim = (wl > 6100.) & (wl < 6900.)
Flux_lim = Flux[mask_lim]
if max(Flux_lim) > 5 * np.mean(Flux_lim):
    max_y_lim = max(Flux_lim) * .9
    #plt.ylim(ymax=max_y_lim)
    # min_y_lim = min(Flux_lim) - 0.2
    # plt.ylim(ymin=min_y_lim,ymax=max_y_lim)

# set Y-axis range (if applicable)
if cmd_args.ymin is not None and cmd_args.ymax is not None:
    plt.ylim(cmd_args.ymin,cmd_args.ymax)
elif cmd_args.ymin is not None:
    plt.ylim(ymin=cmd_args.ymin)
elif cmd_args.ymax is not None:
    plt.ylim(ymax=cmd_args.ymax)

#plt.ylim(ymin=-50.0,ymax=200)
#ax.set(xlabel='Wavelength $(\AA)$')
#ax.set(ylabel=r'F$(\mathrm{10^{-15} erg\ s^{-1} cm^{-2} \AA^{-1}})$')
# set labels and font size
ax.set_xlabel('Wavelength $(\AA)$', fontsize = 32)
ax.set_ylabel(r'F$(\mathrm{10^{-15} erg\ s^{-1} cm^{-2} \AA^{-1}})$', fontsize = 32)
Flux /=1e-15

# if max(Flux) > 9 * np.mean(Flux):
#     max_y_lim = max(Flux) * .9
#     min_y_lim = min(Flux) 
#     plt.ylim(ymin=min_y_lim,ymax=max_y_lim)
ax.plot(wl, Flux, c = "gray", linewidth=1.3, alpha=0.6, zorder=5)
for wl1, mag, magErr, colors, marker_ in zip(wl_br, mag_br, err_br, color_br, marker_br): #
    F = (10**(-(mag + 2.41) / 2.5)) / wl1**2
    F /= 1e-15
    #F *= factor
    ax.scatter(wl1, F, color = colors, marker=marker_, facecolors="none", s=200, zorder=4)
    ax.errorbar(wl1, F, yerr=magErr, fmt='r', marker=None, linestyle=(0, (5, 5)), color=colors, ecolor=colors, elinewidth=3.9, markeredgewidth=3.2, capsize=10)

for wl1_nr, mag_nr, magErr_nr, colors_nr, marker_nrr in zip(wl_nr, mag_nr, err_nr, color_nr, marker_nr): #
    F_nr = (10**(-(mag_nr + 2.41) / 2.5)) / wl1_nr**2
    F_nr /= 1e-15
    #F *= factor
    ax.scatter(wl1_nr, F_nr, color = colors_nr, marker=marker_nrr, s=180, zorder=4)
    ax.errorbar(wl1_nr, F_nr, yerr=magErr_nr, fmt='r', marker=None, linestyle=(0, (5, 5)), color=colors_nr, ecolor=colors_nr, elinewidth=3.9, markeredgewidth=3.2, capsize=10)
#ax.axvline(4686, color='r', linewidth=0.3, linestyle='-', zorder = 6, label="He II")
# plt.text(0.70, 0.19, table["ID"].split("R3.")[-1]).replace(".", "-"),
#              transform=ax.transAxes, fontsize=25, weight='bold')

if cmd_args.ymax is not None:
    ax.annotate(table["ID"].split("R3.")[-1].replace(".", "-"), xy=(9000, 0.415*cmd_args.ymax),  xycoords='data', size=13,
                    xytext=(-120, -60), textcoords='offset points', 
                    bbox=dict(boxstyle="round4,pad=.5", fc="0.94"),)
        
    ax.annotate("r=" + format(float(table["r_PStotal"]), '.2f'), xy=(9000, 0.35*cmd_args.ymax),  xycoords='data', size=13,
            xytext=(-120, -60), textcoords='offset points', 
            bbox=dict(boxstyle="round4,pad=.5", fc="0.94"),)
else:
    None

# plt.annotate(r"H$\alpha$", xy=(7600, 1.09),  xycoords='data', size=23,
#             xytext=(-120, -60), textcoords='offset points', 
#             bbox=dict(boxstyle="round4, pad=.5", fc="#CC0066", alpha=0.7),)
# ax.axvline(4686+65, color='r', linewidth=0.5, linestyle='-', label="He II")
# ax.axvline(6560.28+65, color='k', linewidth=0.5, linestyle='--', label=r"H$\alpha$")
# ax.axvline(5000.7+65, color='k', linewidth=0.5, linestyle='-', label="[O III]")
#ax.legend()
plt.tight_layout()
asciifile = file_spec.replace(".fits", 
                  "-"+i["ID"].split("R3.")[-1].replace(".", "-")+".pdf")

#path_save = "../SDSS-spectra-paper/"
#path_save = "../../paper/Figs2/"
#plt.savefig(os.path.join(datadir_sdss, asciifile))
plt.savefig(asciifile)

