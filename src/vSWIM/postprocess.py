#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:05:46 2026

@author: mrutala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import multiprocessing as mp

from scipy.stats import circmean, circstd
from joblib import Parallel, delayed 

rng = np.random.default_rng()
n_draws = 1000
def clockAngle(mu_b_y, sigma_b_y, mu_b_z, sigma_b_z):
    
    # # Full simulation
    # dist_b_y = rng.normal(loc=mu_b_y, scale=sigma_b_y, size=(n_draws,len(mu_b_y)))
    # dist_b_z = rng.normal(loc=mu_b_z, scale=sigma_b_z, size=(n_draws,len(mu_b_z)))
    
    # dist_clockAngle = np.arctan2(dist_b_y, dist_b_z)
    # mu_clockAngle = circmean(dist_clockAngle, axis=0, low=-np.pi, high=+np.pi)
    # sigma_clockAngle = circstd(dist_clockAngle, axis=0, low=-np.pi, high=+np.pi)
    
    # Error propagation
    mu_clockAngle = np.arctan2(mu_b_y, mu_b_z)
    term1 = (1 / (mu_b_z * (1 + (mu_b_y/mu_b_z)**2)))**2
    term2 = sigma_b_y**2 + ((-mu_b_y/mu_b_z)**2) * sigma_b_z**2
    sigma_clockAngle = np.sqrt(term1 * term2)
    
    return mu_clockAngle, sigma_clockAngle

def coneAngle(mu_b_x, sigma_b_x, mu_b, sigma_b):
    
    # # Full simulation
    # dist_b_x = rng.normal(loc=mu_b_x, scale=sigma_b_x, size=(n_draws,len(mu_b_x)))
    # dist_b = rng.normal(loc=mu_b, scale=sigma_b, size=(n_draws,len(mu_b)))
    
    # dist_coneAngle = np.arccos(dist_b_x, dist_b)
    # mu_coneAngle = circmean(dist_coneAngle, axis=0, low=0, high=+np.pi)
    # sigma_coneAngle = circstd(dist_coneAngle, axis=0, low=0, high=+np.pi)
    
    # Error propagation
    mu_coneAngle = np.arccos(mu_b_x / mu_b)
    term1 = (1 / np.sqrt(mu_b**2 - mu_b_x**2))**2
    term2 = sigma_b_x**2 + ((mu_b_x/mu_b)**2)*sigma_b**2 - 2*sigma_b_x*sigma_b*(mu_b_x/mu_b)*np.corrcoef(mu_b_x, mu_b)[0,1]
    sigma_coneAngle = np.sqrt(term1 * term2)

    return mu_coneAngle, sigma_coneAngle

def postprocessIMF(file):
    
    # Identify needed columns
    clock_cols = ['mu_b_y_SW', 'sigma_b_y_SW', 'mu_b_z_SW', 'sigma_b_z_SW']
    cone_cols  = ['mu_b_x_SW', 'sigma_b_x_SW', 'mu_b_mag_SW', 'sigma_b_mag_SW']
    
    # Read file with only needed columns
    df = pd.read_csv(file, index_col=0) #, usecols=[*clock_cols, *cone_cols])
    
    clockAngle_results = clockAngle(*df[clock_cols].to_numpy().T)
    df['mu_b_clock_angle_SW'] = np.rad2deg(clockAngle_results[0]) % 360
    
    sigma_b_clock_angle_SW = np.rad2deg(clockAngle_results[1])
    bad_index = sigma_b_clock_angle_SW > 360
    sigma_b_clock_angle_SW[bad_index] = 360
    df['sigma_b_clock_angle_SW'] = sigma_b_clock_angle_SW
    df.loc[:, 'bad_b_clock_angle_SW'] = False
    df.loc[bad_index, 'bad_b_clock_angle_SW'] = True
    
    coneAngle_results = coneAngle(*df[cone_cols].to_numpy().T)
    mu_b_cone_angle_SW = np.rad2deg(coneAngle_results[0]) % 180
    mu_b_cone_angle_SW[np.isnan(mu_b_cone_angle_SW)] = 0.0
    df['mu_b_cone_angle_SW'] = mu_b_cone_angle_SW
    sigma_b_cone_angle_SW = np.rad2deg(coneAngle_results[1])
    bad_index = sigma_b_cone_angle_SW > 180
    sigma_b_cone_angle_SW[bad_index] = 180
    df['sigma_b_cone_angle_SW'] = sigma_b_cone_angle_SW
    df.loc[:, 'bad_b_cone_angle_SW'] = False
    df.loc[bad_index, 'bad_b_cone_angle_SW'] = True
    
    # # Only if simulating...
    # # Split into chunks
    # chunk_size = 2e3
    # chunkdf_list = np.array_split(df, len(df)//chunk_size)
    
    # clockAngle_results = Parallel(n_jobs=4)(delayed(clockAngle)(*chunkdf[clock_cols].to_numpy().T) for chunkdf in tqdm.tqdm(chunkdf_list))
    # mu_clockAngle = np.concatenate([r[0] for r in clockAngle_results])
    # sigma_clockAngle = np.concatenate([r[1] for r in clockAngle_results])
    # df['mu_b_clock_angle_SW'] = mu_clockAngle
    # df['sigma_b_clock_angle_SW'] = sigma_clockAngle
    
    # coneAngle_results = Parallel(n_jobs=4)(delayed(coneAngle)(*chunkdf[cone_cols].to_numpy().T) for chunkdf in tqdm.tqdm(chunkdf_list))
    # mu_coneAngle = np.concatenate([r[0] for r in coneAngle_results])
    # sigma_coneAngle = np.concatenate([r[1] for r in coneAngle_results])
    # df['mu_b_cone_angle_SW'] = mu_coneAngle
    # df['sigma_b_cone_angle_SW'] = sigma_coneAngle
    
    df.to_csv(file, float_format='%.3f')
    
    return




# x_clock = np.arange(0,360,1)
# y_clock = np.zeros(x_clock.shape)
# x_cone = np.arange(0,180,1)
# y_cone = np.zeros(x_cone.shape)
# count = 0
# for index, row in tqdm.tqdm(df_hires.query("gap >=0.1").iterrows(), total=len(df_hires.query("gap >=0.1"))):
#     if not row.isna().any():
#         y_clock += scipy.stats.norm.pdf(x_clock, loc=row['mu_b_clock_angle_SW'], scale=row['sigma_b_clock_angle_SW'])
#         y_cone += scipy.stats.norm.pdf(x_cone, loc=row['mu_b_cone_angle_SW'], scale=row['sigma_b_cone_angle_SW'])
#         count += 1
# y_clock /= count
# y_cone /= count

# fig, axs = plt.subplots(ncols=2, figsize=(6,4))

# _ = axs[0].hist(df_hires.query("gap < 0.1")['mu_b_clock_angle_SW'], bins=np.arange(0, 360, 5), color='xkcd:royal blue', alpha=0.5, density=True)
# _ = axs[0].hist(df_hires.query("gap >= 0.1")['mu_b_clock_angle_SW'], bins=np.arange(0,360,5), color='xkcd:coral', alpha=0.5, density=True)
# _ = axs[0].plot(x_clock, y_clock, color='black', lw=1.5)

# _ = axs[1].hist(df_hires.query("gap < 0.1")['mu_b_cone_angle_SW'], bins=np.arange(0,180,5), color='xkcd:royal blue', alpha=0.5, density=True, label='time-to-measurement < 0.1 day')
# _ = axs[1].hist(df_hires.query("gap >= 0.1")['mu_b_cone_angle_SW'], bins=np.arange(0,180,5), color='xkcd:coral', alpha=0.5, density=True, label='time-to-measurement > 0.1 day')
# axs[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode="expand", borderaxespad=0.)

# axs[0].set(xlabel=r'Clock Angle ($atan(B_y/B_z)$) [$^\circ$]', xticks=[0,90,180,270,360], xlim=[0,360], ylabel=r'Measurement Density [arb.]')
# axs[1].set(xlabel=r'Cone Angle ($acos(B_x/|B|)$) [$^\circ$]', xticks=[0,45,90,135,180], xlim=[0,180])