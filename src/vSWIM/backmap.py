#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:51:27 2026

@author: mrutala
"""
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
import pandas as pd
import tqdm 
from scipy import stats
import dill

# import sys
# sys.path.append('/Users/mrutala/projects/vSWIM/src/')
# import vSWIM

average_mars_sw_dict = amsd = {'u_m_mu': np.float64(305.5525212533193),
                               'u_m_sigma': np.float64(130.7325249340937),
                               'u_m_alpha': np.float64(6.682569550462867),
                               'u_x_mu': np.float64(-304.47965038888606),
                               'u_x_sigma': np.float64(129.43818614426067),
                               'u_x_alpha': np.float64(-6.682152656855967),
                               'u_y_mu': np.float64(22.056972284529778),
                               'u_y_sigma': np.float64(17.896448006389015),
                               'u_z_mu': np.float64(0.009161803356804542),
                               'u_z_sigma': np.float64(16.935863122613853)}

# Load the MAVEN-data-driven Solar Wind KDE model
with open('/Users/mrutala/projects/vSWIM/data/MAVEN_derived_mars_solar_wind_kde_model.pkl', 'rb') as f:
    sw_velocity_kde = dill.load(f)

# Generate a skewnormal distribution of solar wind speeds
sw_speed_dist = stats.skewnorm(average_mars_sw_dict['u_m_alpha'], 
                               average_mars_sw_dict['u_m_mu'], 
                               average_mars_sw_dict['u_m_sigma'])

def velocity_from_velocity(df, n_samples):
    samples = np.zeros((df.shape[0], 3, n_samples))
    for i, row in df.reset_index().iterrows():
        d = stats.multivariate_normal(
            mean=[row['v_x'], row['v_y'], row['v_z']], 
            cov=[row['v_x_sigma'], row['v_y_sigma'], row['v_z_sigma']]
            )
        samples[i,:,:] = d.rvs(n_samples).T
        
    return samples

def velocity_from_speed(df, n_samples):
    # n_samples = 1_000
    n_batch = 10_000
    n_max_iter = 10_000
    
    
    samples = np.zeros((df.shape[0], 3, n_samples))
    for i, row in df.reset_index().iterrows():
        # The target distribution is the described by the row
        target_dist = stats.norm(row['v_mag'], row['v_mag_sigma'])
        
        # Get a range of possible speeds to calibrate the max of the target pdf
        possible_speeds = np.linspace(0, 2_000, n_samples*100) #np.linspace(sw_speed_dist.ppf(1e-6), sw_speed_dist.ppf(1-1e-6), n_samples * 100)
        target_pdf_max = target_dist.pdf(possible_speeds).max() * 1.05
        
        # Rejection Sampling
        _samples = []
        j = 0
        while len(_samples) < n_samples:
            # Catch very long loops
            if j == n_max_iter:
                print("Unable to generate sample from rejection sampling)")
                breakpoint()
                
            # Draw candidate (x, y) values
            candidate_X = sw_velocity_kde.resample(n_batch)
            candidate_x = np.sqrt(np.sum(candidate_X**2, axis=0))
            candidate_y = stats.uniform(0, target_pdf_max).rvs(n_batch)
            
            # Keep values for which our candidate_y is less than the pdf y value
            # (e.g., is under the curve of the pdf plot)
            keep_index = candidate_y < target_dist.pdf(candidate_x)
            _samples.extend(candidate_X.T[keep_index])
            
            j+=1
            
        _samples = np.array(_samples)
        if len(_samples) > n_samples:
            # Randomly downsample
            rng = np.random.default_rng()
            _samples = rng.choice(_samples, n_samples, replace=False)
        
        samples[i, :, :] = _samples.T
        
    return samples

def velocity_from_stats(df, n_samples):
    samples = np.zeros((df.shape[0], 3, n_samples))
    # This is independent of row, so just do 10_000ish at a time
    row_indices = np.arange(0, df.shape[0], 1)
    n_chunks = int(df.shape[0]/10_000) 
    # if df.shape[0] > 0:
    #     with tqdm.tqdm(total=df.shape[0]) as pbar:
    #         for sub_row_indices in np.array_split(row_indices, n_chunks):
    #             n_sub_rows = len(sub_row_indices)
    #             flat_samples = sw_velocity_kde.resample(n_sub_rows * n_samples)
    #             samples[sub_row_indices,:,:] = np.moveaxis(flat_samples.reshape(3, n_sub_rows, n_samples), 0, 1)
    #             pbar.update(n_sub_rows)
    if df.shape[0] > 0:
        for sub_row_indices in np.array_split(row_indices, n_chunks):
            n_sub_rows = len(sub_row_indices)
            flat_samples = sw_velocity_kde.resample(n_sub_rows * n_samples)
            samples[sub_row_indices,:,:] = np.moveaxis(flat_samples.reshape(3, n_sub_rows, n_samples), 0, 1)
    
    return samples

def backmap(df, x_plane=0):
    
    # How many samples to draw from velocity distributions, where appropriate
    n_samples = 500 # 750
    # # 750 samples yields ~10% uncertainties on the standard deviation
    # confidence = 0.95
    # alpha = 1 - confidence
    # lower_chi = stats.chi2.ppf(1 - alpha/2, n_samples-1)
    # upper_chi = stats.chi2.ppf(alpha/2, n_samples-1)
    # ci_width = np.sqrt((n_samples-1) / upper_chi) - np.sqrt((n_samples-1) / lower_chi)
    
    # Get ~10,000 element long chunks
    # n_chunks = xyzt.shape[0] // 1e4
    # if n_chunks == 0: n_chunks = 1
    # statevectors = np.array_split(np.hstack([xyzt, v_xyzm]), n_chunks)
    
    # Return a function which returns (N, 1) or (N, M) for vx, vy, vz
    # for N points and M samples (or 1 sample where standard dev. = 0)
    
    # row_is_nan = list(df.loc[:, ['v_mag', 'v_x', 'v_y', 'v_z']].isna().any(axis=0))
    # match row_is_nan:
    #     case [False, False, False, False]:
    #         # Full velocity data
    #         sampler = velocity_from_velocity
    #     case [False, True, True, True]:
    #         # Only speed
    #         sampler = velocity_from_speed
    #     case _:
    #         # No velocity data
    #         sampler = velocity_from_stats
    #         # # Get a 3D velocity distribution from MAVEN stats
    #         # dist = velocity_from_stats()
    #         # def sample(sv):
    #         #     res = dist.resample(size=sv.shape[0]*n_samples)
    #         #     vx = res[0].reshape(sv.shape[0], n_samples)
    #         #     vy = res[1].reshape(sv.shape[0], n_samples)
    #         #     vz = res[2].reshape(sv.shape[0], n_samples)
    #         #     return vx, vy, vz
           
    #         pass
    
    # Δxyzt = np.zeros([df.shape[0], 4])
    # σxyzt = np.zeros([df.shape[0], 4])
    apply_velocity_from_velocity = np.sum(df.loc[:, ['v_mag', 'v_x', 'v_y', 'v_z']].isna() == [False, False, False, False], axis=1) == 4
    apply_velocity_from_speed = np.sum(df.loc[:, ['v_mag', 'v_x', 'v_y', 'v_z']].isna() == [False, True, True, True], axis=1) == 4
    apply_velocity_from_stats = np.sum(df.loc[:, ['v_mag', 'v_x', 'v_y', 'v_z']].isna() == [True, True, True, True], axis=1) == 4
    
    
    samples = np.zeros((df.shape[0], 3, n_samples))
    samples[apply_velocity_from_velocity, :, :] = velocity_from_velocity(df.loc[apply_velocity_from_velocity, :], n_samples)
    samples[apply_velocity_from_speed, :, :] = velocity_from_speed(df.loc[apply_velocity_from_speed, :], n_samples)
    samples[apply_velocity_from_stats, :, :] = velocity_from_stats(df.loc[apply_velocity_from_stats, :], n_samples)
    
    # breakpoint()
    
    # samples = sampler(df, n_samples)
    Δx = x_plane - df['x'].to_numpy()
    Δt_samples = Δx[:,None] / samples[:,0,:]
    Δy_samples = Δt_samples * samples[:,1,:]
    Δz_samples = Δt_samples * samples[:,2,:]
    
    Δxyzt = np.hstack([Δx[:,None], Δy_samples.mean(axis=1)[:,None], Δz_samples.mean(axis=1)[:,None], Δt_samples.mean(axis=1)[:,None]])
    σxyzt = np.hstack([np.zeros((df.shape[0],1)), Δy_samples.std(axis=1)[:,None], Δz_samples.std(axis=1)[:,None], Δt_samples.std(axis=1)[:,None]])
    
    # for index, row in tqdm.tqdm(df.reset_index().iterrows(), total=len(df)):
    #     # dist = stats.multivariate_normal(mean=[row['v_x'], row['v_y'], row['v_z']], cov=[row['u_x_sigma'], row['u_y_sigma'], row['u_z_sigma']])
    #     samples = sampler(row, n_samples)
        
    #     # Get the distance to the x_plane, and the time it takes to get there
    #     # Time can be negative here
    #     Δx = x_plane - row['x']
    #     Δt_samples = Δx / samples[0,:]
        
    #     # In that time, how much have y and z changed?
    #     Δy_samples = Δt_samples * samples[1,:]
    #     Δz_samples = Δt_samples * samples[2,:]
        
        
    #     Δxyzt[index] = Δx, Δy_samples.mean(), Δz_samples.mean(), Δt_samples.mean()
    #     σxyzt[index] = 0., Δy_samples.std(), Δz_samples.std(), Δt_samples.std()
    
    return Δxyzt, σxyzt

def make_planar(df, delta_xyzt, sigma_delta_xyzt):
    
    df = df.copy()
    
    df['x_mu'] = df['x'] + delta_xyzt[:,0]
    df['x_sigma'] = sigma_delta_xyzt[:,0]
    
    df['y_mu'] = df['y'] + delta_xyzt[:,1]
    df['y_sigma'] = sigma_delta_xyzt[:,1]
    
    df['z_mu'] = df['z'] + delta_xyzt[:,2]
    df['z_sigma'] = sigma_delta_xyzt[:,2]
    
    df['mjd_mu'] = df['mjd'] + delta_xyzt[:,3]/(24*60*60)
    df['mjd_sigma'] = sigma_delta_xyzt[:,3]/(24*60*60)
    df['datetime_mu'] = Time(df['mjd_mu'], format='mjd').datetime
    df['datetime_sigma'] = TimeDelta(sigma_delta_xyzt[:,3], format='sec')
    
    df.drop(columns=['x', 'y', 'z', 'datetime', 'mjd'], inplace=True)
    
    return df
    
    