#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:17:07 2026

@author: mrutala
"""


import pandas as pd
import numpy as np
from astropy.time import Time
import astropy.units as u
import datetime as dt
from pathlib import Path

r_M = 3396.19

# Run this locally 
_home_dir = Path(__file__).resolve().parent.parent
def constants():
    
    # Get the home directory of vSWIM
    # home_dir = Path(__file__).resolve().parent.parent
    
    # Generate absolute paths from home
    d = {'src_dir':     _home_dir / 'src',
         'data_dir':    _home_dir / 'data',
         'results_dir': _home_dir / 'results'
         }
    
    # If any of these directories do not exist, make them
    for path in d.values():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            
    return d

def MAVEN(save = False, sigmas=None):
    
    '''
    Downloads data from Jasper Halekas' merged data product online at:
    https://homepage.physics.uiowa.edu/~jhalekas/drivers/drivers_merge_l2_hires.txt
    
    See Halekas et al., 2015, Connerney et al. 2015 for original instrument papers.
    See Halekas et al., 2017 for relevant information on the original merged product. 

    Defaults to not saving file on local drive. 
    
    If save file is enabled, will save to ./Data/drivers_merge_l2_hires.txt on local file
    diretory. 
    '''
    
    # Expected column names
    colNames = ['date_str', 
                'n_p', 'n_a',
                'v_mag', 'v_x', 'v_y', 'v_z', 
                't_p', 
                'b_x', 'b_y', 'b_z']
    
    # Enable offline runs: check if a downloaded file exists, 
    # and read that instead if available
    filename = 'halekas2017_drivers_merge_l2_hires.csv'
    filepath = constants()['data_dir'] / filename
    url = 'https://homepage.physics.uiowa.edu/~jhalekas/drivers/drivers_merge_l2_hires.txt'
    if filepath.exists():
        maven = pd.read_csv(filepath, header=0)
    else:
        maven = pd.read_csv(url, names = colNames, index_col = False, sep = r'\s+')
    if save:
        data_dir = constants()['data_dir']
        
        print("Saving data from https://homepage.physics.uiowa.edu/~jhalekas/drivers.html, see Halekas et al., 2017 to {}.".format(data_dir))
        maven.to_csv(data_dir / 'halekas2017_drivers_merge_l2_hires.csv', index=False)
    
    # !!!! DELETE DOWNLOADS, REDOWNLOAD, THEN REMOVE TRY/EXCEPTs
    # Postprocess
    try: maven['datetime'] = pd.to_datetime(maven['date_str']) 
    except: pass
    try: maven = maven.drop(columns=['n_a'])
    except: pass
    try: maven['b_mag'] = np.sqrt(maven.b_x**2.0 + maven.b_y**2.0 + maven.b_z**2.0)
    except: pass
    try: maven['v_mag'] = np.sqrt(maven.v_x**2.0 + maven.v_y**2.0 + maven.v_z**2.0)
    except: pass
    try: maven['mjd'] = Time(maven['datetime']).mjd
    except: pass
    
    # Add errors if provided
    if sigmas is not None:
        for key, val in sigmas.items():
            maven[key] = val
            
    return maven

def MEX(save = False, maximum_flag = 0, sigmas=None):
    
    '''
    Downloads data from MEX Solar Wind Moment file
    https://archives.esac.esa.int/psa/ftp/MARS-EXPRESS/ASPERA-3/MEX-SUN-ASPERA3-4-SWM-V1.0/DATA/ASP3_IMA_SWM.TAB
    
    Please acknowledge the Principal Investigator(s) as well as the ESA 
    Planetary Science Archive when making a publication using the data you are 
    going to download.
    Principal Investigator(s): R. Lundin (Swedish Institute of Space Science, Kiruna, Sweden)

    Defaults to not saving file on local drive. 
    
    If save file is enabled, will save to ./Data/drivers_merge_l2_hires.txt on local file
    diretory. 
    '''
    
    # Expected column names
    colNames = ['date_str', 
                'n_p', 
                'v_mag',
                't_p', 
                'flag']
    
    # Enable offline runs: check if a downloaded file exists, 
    # and read that instead if available
    filename = 'mex-sun-aspera3-4-swm.csv'
    filepath = constants()['data_dir'] / filename
    url = 'https://archives.esac.esa.int/psa/ftp/MARS-EXPRESS/ASPERA-3/MEX-SUN-ASPERA3-4-SWM-V1.0/DATA/ASP3_IMA_SWM.TAB'
    if filepath.exists():
        df = pd.read_csv(filepath, header=0)
    else:
        df = pd.read_csv(url, names = colNames, index_col = False, sep = r'\s+')
    if save:
        print("Saving data to {}.".format(filepath))
        df.to_csv(filepath, index=False)
        
    # Postprocess
    df['datetime'] = pd.to_datetime(df['date_str'])

    df.loc[:, ['v_x', 'v_y', 'v_z']] = np.nan
    df.loc[:, ['b_x', 'b_y', 'b_z', 'b_mag']] = np.nan
    df['mjd'] = Time(df['datetime']).mjd
    
    
    
    df = df.query("flag <= @maximum_flag").reset_index(drop=True)
    
    # Add errors if provided
    if sigmas is not None:
        for key, val in sigmas.items():
            df[key] = val
    
    return df

def MEX_AMDA(minimum_flag = 0.6, sigmas=None):
    
    '''
    
    '''
    
    # Expected column names
    colNames = ['date_str', 
                'n_p', 
                'v_mag',
                'flag']
    
    # Enable offline runs: check if a downloaded file exists, 
    # and read that instead if available
    filename = 'mex-sw-AMDA-IRAP.txt'
    filepath = constants()['data_dir'] / filename
    # url = 'https://archives.esac.esa.int/psa/ftp/MARS-EXPRESS/ASPERA-3/MEX-SUN-ASPERA3-4-SWM-V1.0/DATA/ASP3_IMA_SWM.TAB'
    if filepath.exists():
        df = pd.read_csv(filepath, comment='#', sep=r'\s+', names=colNames)
    else:
        print("Downloading is not currently supported!")
        return
    # if save:
    #     print("Saving data to {}.".format(filepath))
    #     df.to_csv(filepath, index=False)
        
    # Postprocess
    df['datetime'] = pd.to_datetime(df['date_str'])

    df.loc[:, ['v_x', 'v_y', 'v_z']] = np.nan
    df.loc[:, ['b_x', 'b_y', 'b_z', 'b_mag']] = np.nan
    df['mjd'] = Time(df['datetime']).mjd
    
    
    
    df = df.query("flag >= @minimum_flag").reset_index(drop=True)
    
    # Add errors if provided
    if sigmas is not None:
        for key, val in sigmas.items():
            df[key] = val
    
    return df

def MAVEN_PDS():
    """
    Preliminary (incomplete) code to pull MAVEN directly from the PDS.
    Useful to apply different postprocessing than Halekas+ 2017
    """
    import xarray as xr
    import cdflib
    import cdflib.xarray
    
    url = 'https://pds-ppi.igpp.ucla.edu/data/maven-swia-calibrated/data/onboard_svy_mom/2025/12/mvn_swi_l2_onboardsvymom_20251201_v02_r01.cdf'
    test = cdflib.xarray.cdf_to_xarray(url)
    
    breakpoint()
    return

def MAVENEphemeris(df, kernel_paths):
    import spiceypy as spice
    
    with spice.KernelPool(kernel_paths):
        ets = spice.datetime2et(df['datetime'])
        maven_eph, _ = spice.spkpos("MAVEN", ets, "MAVEN_MSO", "NONE", "MARS")
        
    df.loc[:, ['x', 'y', 'z']] = maven_eph
    return df


def MEXEphemeris(df, kernel_paths):
    import spiceypy as spice
    
    with spice.KernelPool(kernel_paths):
        ets = spice.datetime2et(df['datetime'])
        eph, _ = spice.spkpos("MARS EXPRESS", ets, "MAVEN_MSO", "NONE", "MARS")
        
    df.loc[:, ['x', 'y', 'z']] = eph
    return df

def get_tianwen1_data():
    paths = ['/Users/mrutala/Data/Tianwen/TW1_KH_March2026_SWOnly_2021.csv', 
             '/Users/mrutala/Data/Tianwen/TW1_KH_March2026_SWOnly_2022.csv',
             '/Users/mrutala/Data/Tianwen/TW1_KH_March2026_SWOnly_2023.csv',
             '/Users/mrutala/Data/Tianwen/TW1_KH_March2026_SWOnly_2024.csv']
    dfs = [pd.read_csv(p, parse_dates=['obt']) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={'obt': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert(None)
    df = df.dropna(axis=0, how='any', subset=['b_x', 'b_y', 'b_z', 'b_mag']) # take every 100th sample
    # df = df.query("chopped_labels == 1")
    df['mjd'] = Time(df['datetime']).mjd
    
    df[['n_p', 'v_mag', 't_p', 'v_x', 'v_y', 'v_z']] = np.nan
    df.rename(columns={'px': 'x', 'py': 'y', 'pz': 'z'}, inplace=True)
    
    # !!!!!! Filter as MEX
    rho = np.sqrt(df['y']**2 + df['z']**2) / r_M
    x = df['x'] / r_M
    x_0, L, eps = 0.64, 2.04, 1.03
    standoff_rho = np.sqrt((eps**2 - 1)*(x - x_0)**2 - 2 * eps * L * (x - x_0) + L**2)
    
    df = df.loc[rho > 1.1*standoff_rho, :]
    
    return df
# tianwen1_data = get_tianwen1_data()


def bowshock_Vignes2000(R):
    # https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/1999GL010703
    R_unitless = R/3390 # Vignes uses r_M = 3390 km
    e, L, x0 = 1.03, 2.04, 0.64
    a = (1 - e**2)
    b = (2 * L * e)
    c = R_unitless**2 - L**2
    res1 = (-b + np.sqrt(b**2 - 4 * a * c))/(2*a) + x0
    # res2 = (-b - np.sqrt(b**2 - 4 * a * c))/(2*a) + x0
    return res1*3390

def Tianwen1_DREAMS(save=False, cadence=2*u.min, sigmas=None, 
                    check_for_updates=False):
    import requests
    from bs4 import BeautifulSoup as bs
    import tqdm
    import multiprocessing
    from joblib import Parallel, delayed
    import pandas as pd
    import polars as pl
    import shutil
    import time
    import zipfile
    import glob
    
    tw1_data_dir = constants()['data_dir'] / 'Tianwen1_DREAMS'
    names=['date', 'sampling_rate', 'b_x', 'b_y', 'b_z', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'flag']
    schema = {'date': pl.Datetime(), 
              'sampling_rate': pl.Float64(), 
              'b_x': pl.Float64(), 'b_y': pl.Float64(), 'b_z': pl.Float64(), 
              'x': pl.Float64(), 'y': pl.Float64(), 'z': pl.Float64(), 
              'roll': pl.Float64(), 'pitch': pl.Float64(), 'yaw': pl.Float64(), 
              'flag':pl.Float64()}
    
    def download_and_zip(data_dir):
        # All possible urls for Tianwen-1 MOMAG data
        urls = ['https://space.ustc.edu.cn/dreams/tw1_momag/?magdata=cal&sr=1&year=2021', 
                'https://space.ustc.edu.cn/dreams/tw1_momag/?magdata=cal&sr=1&year=2022', 
                'https://space.ustc.edu.cn/dreams/tw1_momag/?magdata=cal&sr=1&year=2023', 
                'https://space.ustc.edu.cn/dreams/tw1_momag/?magdata=cal&sr=1&year=2024',
                'https://space.ustc.edu.cn/dreams/tw1_momag/?magdata=cal&sr=1&year=2025',
                'https://space.ustc.edu.cn/dreams/tw1_momag/?magdata=cal&sr=1&year=2026',
                ]
        # The url prefix for the actual files to be downloaded
        file_url = 'https://space.ustc.edu.cn/dreams/tw1_momag/fetch.php?datafile='

        # Get a list of remote filepaths (URLs)
        remote_filepaths = []
        for url in urls:
            response = requests.get(url)
            soup = bs(response.text, "html.parser")
            file_links = soup.find_all('a')
            for fl in file_links:
                if fl.string is not None:
                    if '.DAT' in fl.string.upper():
                        remote_filepaths.append(file_url + '/' + fl.string)

        # 
        if len(remote_filepaths) > 0:
            # Construct local filepaths from remote
            data_dir = constants()['data_dir'] / 'Tianwen1_DREAMS'
            data_dir.mkdir(exist_ok=True)
            local_filepaths = [data_dir / Path(rfp).with_suffix('.zip').name for rfp in remote_filepaths]
                
            def _download(remote_filepath, local_filepath, max_attempts=5):
                
                for attempt in range(max_attempts):
                    with requests.get(remote_filepath, stream=True) as response:
                        try:
                            response.raise_for_status()
                            remote_size = int(response.headers.get('Content-Length'))
                            
                            # Get the unzipped local file size
                            if local_filepath.is_file():
                                local_size = sum(f.file_size for f in zipfile.ZipFile(local_filepath).infolist())
                            else:
                                local_size = 0
                            
                            if remote_size != local_size:
                                with open(local_filepath.with_suffix('.dat'), "wb") as file:
                                    for chunk in response.iter_content(chunk_size=4*1024):
                                        if chunk:
                                            file.write(chunk)
                                            
                                with zipfile.ZipFile(local_filepath, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                                    z.write(local_filepath.with_suffix('.dat'), arcname=local_filepath.with_suffix('.dat').name)
                            
                                local_size = sum(f.file_size for f in zipfile.ZipFile(local_filepath).infolist())
                                
                            if remote_size == local_size:
                                break
                            
                        except:
                            time.sleep(10)
                
                # Cleanup
                if local_filepath.with_suffix('.dat').is_file():
                    local_filepath.with_suffix('.dat').unlink()    
                
                return
            
        n_cpus = multiprocessing.cpu_count() - 1
        gen = tqdm.tqdm(
            Parallel(n_jobs=n_cpus, return_as="generator")
            (delayed(_download)(rfp, lfp) for rfp, lfp in zip(remote_filepaths, local_filepaths)),
            total=len(remote_filepaths), desc="Getting Tianwen-1 Data"
            )
        _ = list(gen)
        
        return
    
    if check_for_updates:
        download_and_zip(tw1_data_dir)
        
    # # Read each .dat, handle NaN strings and parse dates to UTC, and save as .csv
    # df = pd.read_csv(local_filepath, comment='#', sep=r'\s+', names=names, parse_dates=['date'], on_bad_lines='skip')
    # df[names[1:]] = df[names[1:]].apply(pd.to_numeric, errors='coerce')
    # df['date'] = df['date'].dt.tz_convert(None)
    # df.to_csv(local_filepath.with_suffix('.csv'), index=False)
    
    def read_all_into_polars(data_dir):
        # Read each zipped .dat file with pandas, then hand off to polars
        colspecs = [(0,26), (30,32), (36,42), (46,52), 
                    (57,62), (66,78), (81,93), (96,108), 
                    (111,118), (121,127), (130,136), (139,141)]
        numeric_cols = list(set(schema.keys()) - {'date'})
        
        def _read(local_filepath):
            def fix_bad_line(bad_line):
                # Things that pandas doesn't like about Tianwen-1 MOMAG:
                #   - Occasionally, the negative sign is offset from the value with spaces
                #   - Occasionally, an errant number makes it's way into a cell that should be NaN
                # This fixes these
                while '-' in bad_line:
                    i = bad_line.index('-')
                    bad_line[i:i+2] = [''.join(bad_line[i:i+2])]
                if len(bad_line) > 12:
                    bad_line = bad_line[0:2] + ['NaN', 'NaN', 'NaN'] + bad_line[-7:]
                return bad_line
            try:
                df = pd.read_csv(local_filepath, comment='#', names=schema.keys(), sep=r'\s{2,}', engine='python', on_bad_lines=fix_bad_line)
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                df['date'] = pd.to_datetime(df['date']).dt.tz_convert(None) # Drop timezone info
            except:
                return local_filepath
                    
            return pl.from_pandas(df, schema_overrides=schema)
        
        local_filepaths = sorted([Path(f) for f in glob.glob((data_dir / '*.zip').as_posix())])
        
        n_cpus = multiprocessing.cpu_count() - 1
        gen = tqdm.tqdm(
            Parallel(n_jobs=n_cpus, return_as="generator")
            (delayed(_read)(lfp) for lfp in local_filepaths),
            total=len(local_filepaths), desc="Parsing Zipped Data"
            )
        res = list(gen)
        df = pl.concat(res)
        
        # Make sure positions are consistent (in km)
        df = df.with_columns(distance_in_km = (pl.col('x').pow(2) + pl.col('y').pow(2) + pl.col('z').pow(2)).sqrt() > 1_000)
        df = df.with_columns(x = pl.when(pl.col('distance_in_km')).then(pl.col('x')).otherwise(pl.col('x') * 3_389.5))
        df = df.with_columns(y = pl.when(pl.col('distance_in_km')).then(pl.col('y')).otherwise(pl.col('y') * 3_389.5))
        df = df.with_columns(z = pl.when(pl.col('distance_in_km')).then(pl.col('z')).otherwise(pl.col('z') * 3_389.5))
        df = df.drop("distance_in_km")
        df = df.sort('date')
        
        # Write to quick IO file
        df.write_ipc(data_dir / 'all_Tianwen-1.arrow')
        
        return data_dir / 'all_Tianwen-1.arrow'
    
    tianwen1_df_path = tw1_data_dir/'all_Tianwen-1.arrow'
    if not tianwen1_df_path.exists():
        tianwen1_df_path = read_all_into_polars(tw1_data_dir)
        
    # Create a solar wind filtered file, or read it if it exists
    def solar_wind_filter(R):
        return bowshock_Vignes2000(R) + 3390*(R/(4.5*3390))**2 + 0.1*3390
    
    
    
    tianwen1_solarwind_df_path = constants()['data_dir'] / 'SW_Tianwen-1.arrow'
    if not tianwen1_solarwind_df_path.exists():
        tianwen1_R = pl.scan_ipc(tianwen1_df_path).with_columns(R = (pl.col('y').pow(2) + pl.col('z').pow(2)).sqrt()).select('R').collect()
        minimum_x = solar_wind_filter(tianwen1_R['R']).rename('minimum_x')
        tianwen1_df = ( 
            pl.scan_ipc(tianwen1_df_path, hive_schema=schema)
            .filter(pl.col('x') >= minimum_x)
            .filter(pl.col('flag') == 0)
            .with_columns(pl.col('date').str.to_datetime())
            .collect()
            )
        tianwen1_df.write_ipc(tianwen1_solarwind_df_path)
    else:
        tianwen1_df = pl.read_ipc(tianwen1_solarwind_df_path)
        
    # Standard formatting etc.
    df = tianwen1_df.to_pandas()
    df = df.rename(columns={'date': 'datetime'})
    df['mjd'] = Time(df['datetime']).mjd
    df['b_mag'] = np.sqrt(np.sum(df[['b_x', 'b_y', 'b_z']]**2, axis=1))
    df[['n_p', 'v_mag', 't_p', 'v_x', 'v_y', 'v_z']] = np.nan
    
    # Add errors if provided
    if sigmas is not None:
        for key, val in sigmas.items():
            df[key] = val
    
    return df

def InstantanousSynodicCarringtonPeriod(df):
    # Get Mars velocity for realtime Carrington period
    from astroquery.jplhorizons import Horizons
    mars = Horizons('499', 
                    epochs={'start': df['datetime'].min().strftime('%Y-%m-%d'), 
                            'stop': (df['datetime'].max() + dt.timedelta(days=1)).strftime('%Y-%m-%d'), 
                            'step': '1d'})
    mars_vectors = mars.vectors().to_pandas()

    # r_vec = np.hstack([mars_vectors[['x', 'y']].to_numpy(), np.zeros((mars_vectors.shape[0],1))])
    # v_vec = np.hstack([mars_vectors[['vx', 'vy']].to_numpy(), np.zeros((mars_vectors.shape[0],1))])
    r_vec = mars_vectors[['x', 'y', 'z']].to_numpy()
    v_vec = mars_vectors[['vx', 'vy', 'vz']].to_numpy()
    
    angular_velocity = np.cross(r_vec, v_vec) / (r_vec**2).sum(axis=1)[:,None]
    angular_velocity_synodic = (2*np.pi)/25.38 - angular_velocity[:,2]
    period_synodic = (2*np.pi) / angular_velocity_synodic
    
    instant_synodic_CRPeriod = np.interp(Time(df['datetime']).jd, 
                                         mars_vectors['datetime_jd'], 
                                         period_synodic
                                         )
    return instant_synodic_CRPeriod
