#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:17:07 2026

@author: mrutala
"""


import pandas as pd
import numpy as np

from pathlib import Path

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

def MAVEN(save = False):
    
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
                'n_p', 'n_alpha', 
                'v_mag', 'v_x', 'v_y', 'v_z', 
                'tp', 
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
    
    # Postprocess
    maven['datetime'] = pd.to_datetime(maven['date_str'])

    maven['b_mag'] = np.sqrt(maven.b_x**2.0 + maven.b_y**2.0 + maven.b_z**2.0)

    maven['v_mag'] = np.sqrt(maven.v_x**2.0 + maven.v_y**2.0 + maven.v_z**2.0)
    
    if save:
        data_dir = constants()['data_dir']
        
        print("Saving data from https://homepage.physics.uiowa.edu/~jhalekas/drivers.html, see Halekas et al., 2017 to {}.".format(data_dir))
        maven.to_csv(data_dir / 'halekas2017_drivers_merge_l2_hires.csv', index=False)

    return maven

def MEX(save = False, maximum_flag = 0):
    
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
                'tp', 
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
    
    # Postprocess
    df['datetime'] = pd.to_datetime(df['date_str'])

    df.loc[:, 'n_alpha'] = np.nan
    df.loc[:, ['v_x', 'v_y', 'v_z']] = np.nan
    df.loc[:, ['b_x', 'b_y', 'b_z', 'b_mag']] = np.nan
    
    if save:
        print("Saving data to {}.".format(filepath))
        df.to_csv(filepath, index=False)
    
    return df.query("flag <= @maximum_flag").reset_index(drop=True)
