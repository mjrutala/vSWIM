#created by A. R. Azari on 2/1/2024
#see BSD-3 Liscense at https://github.com/abbyazari/vSWIM/blob/main/LICENSE.md
#see relevant materials at https://github.com/abbyazari/vSWIM/blob/main/Citation.bib 
#if using the original MAVEN generated data see https://github.com/abbyazari/vSWIM/tree/main
#for original data citations including but not limited to: 
#Halekas et al., 2017, Halekas et al., 2015, Connerney et al., 2015

#import required data grabbing and storing packages
import requests
import glob
import os
import regex                            as     re

#import GPU enabled GP packages
import tensorflow              as     tf
import tensorflow_probability  as     tfp
import gpflow

#import useful analysis
from   scipy.spatial.distance  import cdist
import pandas                  as     pd
import numpy                   as     np
import datetime                as     dt
from   sklearn.preprocessing   import StandardScaler, MinMaxScaler
import matplotlib.pyplot       as     plt

import getData
from astropy.time import Time
from joblib import Parallel, delayed
import multiprocessing

import time
import tqdm
import unittest

#set random numbers for consistency between this run and 
#future use
rndm_no = 42
np.random.seed(rndm_no) #set numpy random seed
tf.random.set_seed(rndm_no)

#set global variable scaling constants for GP
maxRescale = 100
subsetSize = 1000
min_l      = 0.0 
mid_l      = 0.1
max_l      = 0.5
init_var   = 3

#full set of solar wind parameters
fullParams = ['b_x',     'b_y',    'b_z',   'b_mag',
              'v_x',     'v_y',    'v_z',   'v_mag', 
              't_p',      'n_p']


def getOrbitalData():
    
    '''
    Downloads MAVEN orbital ephemeris data from NASA SPICE. Optional, will only run if needing orbit numbers.
    '''

    baseURL = 'https://naif.jpl.nasa.gov/pub/naif/MAVEN/kernels/spk/'

    r = requests.get(baseURL)

    x = re.findall('"(maven_orb_rec_.*.orb)"', r.text)

    x.sort()

    orbs     = np.zeros(0)
    apoDates = np.zeros(0)


    for link in x:

        r = requests.get(baseURL+link)

        for line in r.text.splitlines()[2:]:

            orb = line.split()[0]

            dateStr = line.split()[6] + line.split()[7]  + line.split()[8]  + '-' + line.split()[9]

            apoDate = dt.datetime.strptime(dateStr, "%Y%b%d-%H:%M:%S")

            orbs = np.append(orbs, np.int32(orb))

            apoDates = np.append(apoDates, apoDate)
            
    return(orbs, apoDates)


def formatSpacecraftData(df):
    
    # Add subset indices
    numSubsets = int(len(df) / subsetSize)
    
    # NOTE: code previously excluded subsets < 1000 elements long
    # This has been removed to allow small dataset usage
    df['SubsetIndex'] = [int(s) for s in np.arange(len(df))/1000]
    if (df['SubsetIndex'] == df['SubsetIndex'].max()).sum() < 500:
        df.loc[df['SubsetIndex'] == df['SubsetIndex'].max(), 'SubsetIndex'] = df['SubsetIndex'].max() - 1

    return df
    

def runvSWIM(startDate = dt.datetime(2015, 1,  1), stopDate  = dt.datetime(2015, 1,  4), cadence = 60*60, 
             params = ['b_x', 'b_y', 'b_z', 'b_mag',
                       'v_x', 'v_y', 'v_z', 'v_mag', 
                       't_p', 'n_p'],    
             getOrb = False, saveModelResults = False, saveSourceData = False, returnOriginal = False, verbose = False,
             sourceData = 'MAVEN'):
    
    '''
    Run the vSWIM model over a set period of time at a cadence in seconds. 
    
    If you want to save a copy of the original MAVEN file to the Data folder
    use saveMAVENData = True.
    
    If you want to save the results of the model to the Data folder
    use saveModelResults = True.
    
    You can run any of the following parameters: b_x, b_y, b_z, b_mag,
                                                 v_x, v_y, v_z, v_mag, 
                                                 t_p,  n_p
                                                 
    Note 1: future improvements will have user input checks.
    
    Note 2: this function can not be run before dt.datetime(2014, 11, 12, 12).
    
    '''
    #check user inputs
    if ((type(startDate)  != type(dt.datetime(2015, 1, 1))) | 
             ((type(stopDate) != type(dt.datetime(2015, 1, 1))))):
        raise TypeError('Check time range type, use dt.datetime format.')
    
    if (type(params) != type(['a', 'b'])):
        raise ValueError("Check solar wind entry parameter type, use ['param1', 'param2'] format.")
        
    #check if user used a real solar wind parameter and correct time range.
    if (startDate >= stopDate):
        raise ValueError("Can not run on stopDate <= startDate.")
    
    for p_i in params:
        if not p_i in fullParams:           
            raise ValueError('{} is not within valid solar wind options: {}'.format(p_i, fullParams))
                
    # If sourceData is a string, read the correct spacecraft; else, use as-is
    if type(sourceData) is str:
        match sourceData.upper():
            case 'MAVEN':
                insitu_df = getData.MAVEN(saveSourceData)
            case 'MEX':
                insitu_df = getData.MEX(saveSourceData)
    else:
        insitu_df = sourceData
        
    insitu_df = formatSpacecraftData(insitu_df)
    
    #check if selected dates are valid
    if ((startDate < insitu_df.datetime.min()) | 
        (stopDate  > insitu_df.datetime.max())):
        raise ValueError(
            'Can only run from {} to {}, pick new time range.'.format(
            insitu_df.date_SW.min(), insitu_df.date_SW.max()))
        
    startSubsets = insitu_df[::subsetSize]
    
    indexStart = startSubsets.loc[startSubsets['datetime'] <= startDate, 'SubsetIndex'].values[-1]
    indexStop  = startSubsets.loc[startSubsets['datetime'] >= stopDate,  'SubsetIndex'].values[0]
    
    results = pd.DataFrame()

    results['date_[utc]']  = pd.to_datetime(np.arange(startDate, stopDate, 
                                                        dt.timedelta(seconds = cadence)))

    results['date_[unix]']  = ((results['date_[utc]'] - pd.Timestamp("1970-01-01")) //
                                pd.Timedelta('1s'))


    results['gap']      = np.nan
    
    if getOrb:
        
        print('Generating MAVEN orbit information.')

        results['orb']          = np.nan

        orbs, apoDates = getOrbitalData()

        orbStart = orbs[(apoDates >= results['date_[utc]'][0])][0] - 1 
        orbStop  = orbs[(apoDates <  results['date_[utc]'][len(results) - 1])][-1] 


        for orb in np.arange(orbStart, orbStop + 1):


            index = (orbs == orb)

            startApo = apoDates[orbs == orb][0] 
            endApo   = apoDates[orbs == orb + 1][0]

            orbIndex = ((results['date_[utc]'] >= startApo) & (results['date_[utc]'] < endApo))

            results.loc[orbIndex, 'orb'] = orb

    print('Running from {} to {}, in {} segments, and for parameters:'.format(startDate, 
                                                                              stopDate, 
                                                                              indexStop - indexStart))

    for p in params:

        print('{}'.format(p))

        results['mu_{}'.format(p)]            = np.nan

        results['sigma_{}'.format(p)]         = np.nan

        results['mu_{}_normed'.format(p)]     = np.nan

        results['sigma_{}_normed'.format(p)]  = np.nan
    
    arrEnum  = np.arange(indexStart, indexStop, 1)
    
    for i, o in enumerate(tqdm.tqdm(arrEnum, desc='Processing segments', position=0)):
        
        data = insitu_df.query("SubsetIndex == @o")
        
        indResults = ((results['date_[utc]'] >= data.datetime.values[0]) & 
                        (results['date_[utc]'] < data.datetime.values[-1]))

        
        # X_train = data['date_SW_unix'].values.reshape(-1, 1) 
        X_train = Time(data['datetime']).mjd[:,None]

        # Sizes of gaps, in days
        dist_matrix = cdist(
            Time(results.loc[indResults, 'date_[utc]']).mjd[:,None], 
            Time(data['datetime']).mjd[:,None])
        results.loc[indResults, 'gap'] = np.around(np.min(dist_matrix, 1), decimals = 3)
        
        for p in tqdm.tqdm(params, desc='Processing [{}]'.format(', '.join(params)), leave=False, position=1):

            # if verbose:
            #     print(p)

            y_train = data['{}'.format(p)].values.reshape(-1, 1)


            normScaler = StandardScaler()

            normScaler.fit(y_train)

            mmScaler = MinMaxScaler(feature_range=(0, maxRescale))
            mmScaler.fit(X_train)

            X_normed_train = mmScaler.transform(X_train)
            y_normed_train = normScaler.transform(y_train)

            dists        = cdist(X_normed_train, X_normed_train)

            dists_noZeros = dists[dists != 0]

            minLength = np.quantile(dists_noZeros, min_l)

            midLength = np.quantile(dists_noZeros, mid_l)

            maxLength = np.quantile(dists_noZeros, max_l)


            signal_kernel = gpflow.kernels.RationalQuadratic(variance = init_var)


            signal_kernel.lengthscales = gpflow.Parameter(midLength, 
                                    transform=tfp.bijectors.SoftClip(
                                        gpflow.utilities.to_default_float(minLength),
                                        gpflow.utilities.to_default_float(maxLength)))


            model = gpflow.models.GPR((X_normed_train, y_normed_train), kernel=signal_kernel)

            opt = gpflow.optimizers.Scipy()
            t0 = time.time()
            opt.minimize(model.training_loss, model.trainable_variables)
            # print("Model trained in {}s".format(time.time() - t0))
            # if verbose: 
            #     gpflow.utilities.print_summary(model) #, "notebook")

            #----------and now save results

            X_model = mmScaler.transform(Time(results.loc[indResults, 'date_[utc]']).mjd.reshape(-1, 1))

            # Replace sampling with embarassingly parallel sampling
            n_cpus = multiprocessing.cpu_count() - 1
            X_model_segments = np.array_split(X_model, n_cpus)
            pred_y = Parallel(n_jobs=n_cpus)(delayed(model.predict_y)(_X) for _X in X_model_segments)
            
            mean_model = np.concatenate([py[0] for py in pred_y])
            var_model = np.concatenate([py[1] for py in pred_y])

            std_model = np.sqrt(var_model)

            results.loc[indResults, 'mu_{}_normed'.format(p)]    = mean_model

            results.loc[indResults, 'sigma_{}_normed'.format(p)] = std_model

            mean_model_unnorm = normScaler.inverse_transform(mean_model.reshape(-1, 1))[:, 0]

            std_model_unnorm  = std_model*normScaler.scale_

            results.loc[indResults, 'mu_{}'.format(p)]    = mean_model_unnorm

            results.loc[indResults, 'sigma_{}'.format(p)] = std_model_unnorm

    if saveModelResults:

        #make directory

        locResults = './results/'

        try:
            os.mkdir(locResults)
        
        except:
          #if directory already exists do nothing  
          pass
          

        print('\nSaving model outputs to {}'.format(locResults))
        
        results.to_csv('{}vSWIM_{:%Y-%m-%d:%H-%M-%S}_{:%Y-%m-%d:%H-%M-%S}.csv'.format(locResults, startDate, stopDate))

    if returnOriginal: 
        
        return insitu_df, results

    else:
        return results


class TestvSWIM(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from sklearn.model_selection import GroupShuffleSplit
        
        mean_gap_size = 0.5 # Days
        test_cadence = 2*60 # s
        test_date = np.arange(dt.datetime(2000, 1, 1), dt.datetime(2000, 4, 1), dt.timedelta(seconds=test_cadence)).astype(dt.datetime)
        test_df = pd.DataFrame({'date_SW': test_date})
        
        # Add unix date and artifical solar wind signal as v_mag_SW
        test_df['date_SW_unix'] = (test_df['date_SW'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        rng = np.random.default_rng()
        test_df['v_mag_SW'] = 400 + 100 * np.sin(np.linspace(0, 2*np.pi, len(test_df))) + 10 * np.sin(np.linspace(0, 20*np.pi, len(test_df))) + rng.normal(0, 1, len(test_df))
        
        # Drop groups of the data to simulate partial solar wind coverage
        test_groups = np.arange(len(test_date)) // (mean_gap_size*24*60*60/test_cadence)
        
        # Split times into groups, then randomly select groups to drop
        gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
        keep_groups, lose_groups = next(gss.split(test_date, groups=test_groups))
        
        # Drop lose_groups in new df
        keep_df = test_df.drop(index=lose_groups)
        keep_df.reset_index(inplace=True, drop=True)
        lose_df = test_df.drop(index=keep_groups)
        lose_df.reset_index(inplace=True, drop=True)
        
        # Add SubsetIndex so test_spacecraft_df can be passed to runvSWIM
        keep_df['SubsetIndex'] = keep_df.index // subsetSize
        
        self.test_df = test_df
        self.keep_df = keep_df
        self.lose_df = lose_df
        
        test_model_df = runvSWIM(startDate=dt.datetime(2000,1,5), stopDate=dt.datetime(2000,3,25), 
                                 cadence=3600, params=['v_mag_SW'], 
                                 spacecraftData = self.keep_df)
        
        self.test_model_df = test_model_df
        
        
        df = lose_df.query("@test_model_df['date_[utc]'].iloc[0] < date_SW < @test_model_df['date_[utc]'].iloc[-1]")
        mu_interp = np.interp(df['date_SW_unix'],test_model_df['date_[unix]'], test_model_df['mu_v_mag_SW'])
        sigma_interp = np.interp(df['date_SW_unix'], test_model_df['date_[unix]'], test_model_df['sigma_v_mag_SW'])
        
        ZScore = (df['v_mag_SW'] - mu_interp) / sigma_interp
        self.ZScore = ZScore
        
        fig, ax = plt.subplots()
        ax.hist(ZScore, bins=np.arange(-4, 4, 0.1))
        ax.axvline(ZScore.mean(), color='black', lw=1)
        ax.axvline(ZScore.mean()+ZScore.std(), color='black', lw=1, ls=':')
        ax.axvline(ZScore.mean()-ZScore.std(), color='black', lw=1, ls=':')
        ax.annotate((r"$\mu$ = {0:.3f}"+"\n"+r"$\sigma$ = {1:.3f}").format(ZScore.mean(), ZScore.std()),
                    (0,1), (1,-1), xycoords='axes fraction', textcoords='offset fontsize', 
                    ha='left', va='top')
        
        if ZScore.mean() < -0.1:
            breakpoint()
        return
        
    def test_mean(self):
        message = "Absolute Mean of the Z-Score > 0.10!"
        self.assertAlmostEqual(self.ZScore.mean(), 0, delta=0.10, msg=message)

        
    def test_std(self):
        message = "Absolute Standadrd Deviation of the Z-Score > 0.10!"
        self.assertAlmostEqual(self.ZScore.std(), 1, delta=0.10, msg=message)

# if __name__ == "__main__":
    
        
#     #add command line functionality
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument("--start_date", type=dt.datetime.fromisoformat, default="2015-01-01", help="start date in any ISO format")
#     parser.add_argument("--end_date", type=dt.datetime.fromisoformat, default="2015-01-04", help="start date in any ISO format")
#     parser.add_argument("--cadence", type=int, default=1, help="interpolation cadence in seconds")
#     parser.add_argument("--params_list", type=str, nargs='+', default = ['v_x_SW'], help="solar wind parameters to interpolate, "+\
#                                                                                          "list any of:  b_x_SW, b_y_SW, b_z_SW, b_mag_SW, "+\
#                                                                                          "v_x_SW, v_y_SW, v_z_SW, v_mag_SW, tp_SW,  np_SW" )

#     parser.add_argument("--get_orb", action="store_false", help="include orbital information")
#     parser.add_argument("--save_model_results", action="store_false", help="save model results to csv in the data folder")
#     parser.add_argument("--save_maven_data", action="store_false", help="save a copy of the original MAVEN data in the data folder")
#     parser.add_argument("--return_original", action="store_false", help="returns a copy of the original MAVEN data to user")
#     parser.add_argument("--verbose", action="store_false", help="increase output verbosity")
    
#     args = parser.parse_args()
