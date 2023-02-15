# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:41:11 2022

@author: st5536
"""

from glob import glob
import pandas as pd
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import matplotlib.colors as mcolors

from py_instruments import read_data as io


def compare_exact(first, second):
    """
    Return whether two dicts of arrays are exactly equal.
    From https://stackoverflow.com/questions/26420911/
    comparing-two-dictionaries-with-numpy-matrices-as-values.
    """
    import numpy as np
    if first.keys() != second.keys():
        return False
    return all(np.array_equal(first[key], second[key]) for key in first)


# %% Read USMPS
data_path = r'\\IMKAAF-SRV1\agm-field\Campaigns\CORONA\USMPS-01\2020_06_1'

file_names = glob(data_path + r'\*.txt')

df_lst = []
attrs_lst = []
for file in file_names:
    df = io.read_usmps_data(file)
    df.drop(['time since start / s', 'duration of interval / s'], axis=1, inplace=True)
    df.columns = df.columns.astype(float)
    attrs_lst.append(df.attrs)
    df.attrs = {}
    df_lst.append(df)

boolean_lst = []
for i, j in permutations(range(len(attrs_lst)), r=2):
    boolean_lst.append(compare_exact(attrs_lst[i], attrs_lst[j]))

usmps = pd.concat(df_lst, axis=0)

if all(boolean_lst):
    usmps.attrs = attrs_lst[0]

# %% Read Fidas
data_path = r'\\IMKAAF-SRV1\agm-field\Campaigns\CORONA\FIDAS-02\L0_Data\2020_06'

file_names = glob(data_path + r'\*.txt')

df_lst = []
attrs_lst = []
for file in file_names:
    df = io.read_fidas_data(file)
    df.drop(['time since start / s', 'duration of interval / s'], axis=1, inplace=True)
    df.columns = df.columns.astype(float)
    attrs_lst.append(df.attrs)
    df.attrs = {}
    df_lst.append(df)

boolean_lst = []
for i, j in permutations(range(len(attrs_lst)), r=2):
    boolean_lst.append(compare_exact(attrs_lst[i], attrs_lst[j]))

fidas = pd.concat(df_lst, axis=0)

if all(boolean_lst):
    fidas.attrs = attrs_lst[0]

# %% Convert dN to dV
fidas_dV = np.pi / 6 * fidas.multiply(fidas.columns**3, axis=1).divide(fidas.attrs['dlogdp'])
usmps_dV = np.pi / 6 * usmps.multiply(usmps.columns**3, axis=1).divide(usmps.attrs['dlogdp'])

fidas_dN_dlogdp = fidas.divide(fidas.attrs['dlogdp'])
usmps_dN_dlogdp = usmps.divide(usmps.attrs['dlogdp'])

# %%
start, stop = pd.to_datetime('2020-06-02 00:00:00'), pd.to_datetime('2020-06-03 00:00:00')
fig, ax = plt.subplots()
ax.plot(usmps_dN_dlogdp.columns, usmps_dN_dlogdp.loc[start:stop, :].mean(axis=0),
        ls='', marker='x')
ax.plot(fidas_dN_dlogdp.columns, fidas_dN_dlogdp.loc[start:stop, :].mean(axis=0),
        ls='', marker='o')
ax.set_xscale('log')
ax.set_yscale('log')

# %%
start, stop = pd.to_datetime('2020-06-02 00:00:00'), pd.to_datetime('2020-06-03 00:00:00')
fig, ax = plt.subplots()
c = ax.pcolormesh(usmps_dN_dlogdp.loc[start:stop, :].index,
                  usmps_dN_dlogdp.loc[start:stop, :].columns,
                  usmps_dN_dlogdp.loc[start:stop, :].T,
                  norm=mcolors.LogNorm())
fig.colorbar(c, ax=ax, label=r'd$N$/dlog$d_\mathrm{p}$ / cm$^{-3}$')
locator = AutoDateLocator()
formatter = ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
