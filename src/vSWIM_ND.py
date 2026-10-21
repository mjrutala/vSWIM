#vSWIM 0.0 created by A. R. Azari on 2/1/2024
#vSWIM 1.0 edited heavily by M. J. Rutala on 2026/9/15
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
import polars                  as pl
import numpy                   as     np
import datetime                as     dt
from   sklearn.preprocessing   import StandardScaler, MinMaxScaler
from   sklearn.pipeline        import Pipeline, FunctionTransformer
import matplotlib.pyplot       as     plt

import getData
import backmap
from astropy.time import Time
from astropy import units as u
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
maxRescale = 10
subsetSize = 1000
subsetSpan = 30*u.day
min_l      = 0.0 
mid_l      = 0.1
max_l      = 0.5
init_var   = 3
yz_res_eff = 1_000 # km

#full set of solar wind parameters
fullParams = ['b_x',     'b_y',    'b_z',   'b_mag',
              'v_x',     'v_y',    'v_z',   'v_mag', 
              't_p',      'n_p']

class KnownNoiseGPR(gpflow.models.GPR):
    """GPR with fixed, per-point noise variances."""

    def __init__(self, data, kernel, noise_var, **kwargs):
        
        super().__init__(data, kernel, noise_variance=1e-5, **kwargs)
        
        gpflow.utilities.set_trainable(self.likelihood.variance, False)  # freeze the tiny jitter term
        self.known_noise = tf.Variable(
            noise_var.reshape(-1), dtype=gpflow.default_float(), trainable=False
        )

    def _noisy_K(self, X):
        noise = self.known_noise + self.likelihood.variance
        return self.kernel(X) + tf.linalg.diag(noise)

    def log_marginal_likelihood(self):
        X, Y = self.data
        L = tf.linalg.cholesky(self._noisy_K(X))
        m = self.mean_function(X)
        return tf.reduce_sum(gpflow.logdensities.multivariate_normal(Y, m, L))

    def maximum_log_likelihood_objective(self):
        return self.log_marginal_likelihood()

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        X, Y = self.data
        err = Y - self.mean_function(X)
        kmn = self.kernel(X, Xnew)
        knn = self.kernel(Xnew, full_cov=full_cov)
        mean, var = gpflow.conditionals.util.base_conditional(
            kmn, self._noisy_K(X), knn, err, full_cov=full_cov, white=False)
        return mean + self.mean_function(Xnew), var


def input_noise_term(model, X, var_x):
    """Per-point  grad f^T diag(var_x) grad f  at the training inputs."""
    Xt = tf.convert_to_tensor(X, dtype=gpflow.default_float())
    with tf.GradientTape() as tape:
        tape.watch(Xt)
        mean, _ = model.predict_f(Xt)
    # mean_i depends only on Xt_i, so the gradient of the sum gives per-point slopes
    grads = tape.gradient(mean, Xt)                      # (N, 2)
    return tf.reduce_sum(grads**2 * var_x, axis=1).numpy()  # (N,)
               
from check_shapes import inherit_check_shapes
class FixedVarianceOfMean(gpflow.functions.Function):
    def __init__(self, Y_var: gpflow.base.AnyNDArray):
        # n_repeats = Y.shape[-1]
        self.var = Y_var

    @inherit_check_shapes
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.var


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

def calculateSubsetIndex(date, startDate, span):
    return np.floor((Time(date).mjd + 0.5*span.to(u.day).value - Time(startDate).mjd)/span.to(u.day).value)

def assignSubsetIndex(df, startDate, span):
    # Adds subset indices based on time, and indexed to the start date
    df['SubsetIndex'] = calculateSubsetIndex(df['datetime'], startDate, span)
    return df

def cluster_by_resolution(arr, res):
    import sklearn
    # Expect NxM for array (where M is features), and M for res
    
    # Scale the input array by desires resolution to get unitless
    scaled_arr = arr / np.array([[*res]])
    
    # Full connectivity agglomerative clustering, chunked by kmeans
    # This may split ideal clusters along chunk edges, 
    # but performs much, much faster
    n_kmeans_clusters = int(np.ceil(scaled_arr.shape[0] / 10_000))
    kmClustering = sklearn.cluster.KMeans(n_clusters = n_kmeans_clusters)
    kmClusters = kmClustering.fit(scaled_arr)
    aggClusterLabels = np.full(scaled_arr.shape[0], -1, dtype=int)
    for supercluster_label in tqdm.tqdm(range(kmClusters.labels_.max()+1)):
        supercluster_index = kmClusters.labels_ == supercluster_label
        
        aggClustering = sklearn.cluster.AgglomerativeClustering(
            distance_threshold=1, compute_full_tree=True, linkage='complete', n_clusters=None)
        aggClusters = aggClustering.fit(scaled_arr[supercluster_index,:])
        
        aggClusterLabels[supercluster_index] = aggClusters.labels_ + aggClusterLabels.max() + 1
    
    return aggClusterLabels

def runvSWIM(startDate = dt.datetime(2015, 1,  1), stopDate  = dt.datetime(2015, 1,  4), cadence = 60*60, 
             params = ['b_x', 'b_y', 'b_z', 'b_mag',
                       'v_x', 'v_y', 'v_z', 'v_mag', 
                       'n_p'],    
             getOrb = False, saveModelResults = False, saveSourceData = False, returnOriginal = False, verbose = False,
             sourceData = ['MAVEN', 'MEX']):
    
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
    
    # # If sourceData is a string, read the correct spacecraft; else, use as-is
    # if type(sourceData) is str:
    #     sourceData = [sourceData]
    # if type(sourceData) is list:
        
    #     for source in sourceData:
    #         match source.upper():
    #             case 'MAVEN':
    #                 insitu_df = getData.MAVEN(saveSourceData)
    #             case 'MEX':
    #                 insitu_df = getData.MEX(saveSourceData)
    # else:
    #     insitu_df = sourceData
    insitu_list = sourceData # !!!! need to fix for string keywords
    
    # I'd love to trim this df, but we need to keep v_x, v_y, v_z even if not requested...
    # all_insitu_df = (pd.concat(insitu_list)
    #                  .sort_values(by='mjd')
    #                  .reset_index(drop=True)
    #                  .loc[:, ['datetime', 'mjd', 'asset', *params, *['{}_sigma'.format(p) for p in params]]]
    #                  )
    
    all_insitu_df = (pd.concat(insitu_list)
                     .sort_values(by='mjd')
                     .reset_index(drop=True)
                     )

    
    #check if selected dates are valid
    startDateValid = all_insitu_df['datetime'].min()
    stopDateValid = all_insitu_df['datetime'].max()
    if ((startDate < startDateValid) | 
        (stopDate  > stopDateValid)):
        raise ValueError(
            'Can only run from {} to {}, pick new time range.'.format(
            startDateValid, stopDateValid))
    
    
    # Assign subset indices based on the MEASUREMENT time
    # subset_list = [assignSubsetIndex(insitu, startDate, span=subsetSpan) for insitu in insitu_list]
    all_insitu_df['SubsetIndex'] = calculateSubsetIndex(all_insitu_df.datetime, startDate, subsetSpan)
    
    # Get the indices we care about and subset for efficiency
    # startSubsets = insitu_df[::subsetSize]
    # indexStart = startSubsets.loc[startSubsets['datetime'] <= startDate, 'SubsetIndex'].values[-1]
    # indexStop  = startSubsets.loc[startSubsets['datetime'] >= stopDate,  'SubsetIndex'].values[0]
    indexStart = 0
    indexStop = int(all_insitu_df.query("datetime >= @stopDate")['SubsetIndex'].values[0])
    # subset_list = [subset.query("@indexStart-1 <= SubsetIndex <= @indexStop+1") for subset in subset_list]
    
    # Combined into one dataframe, with labels for original source
    all_insitu_df = all_insitu_df.query("@indexStart-1 <= SubsetIndex <= @indexStop+1")
    
    # Backmap *everything* here, being careful about memory limits
    all_planar_df = backmap.make_planar(all_insitu_df, *backmap.backmap(all_insitu_df, x_plan=3*3396.19))
    
    # Initialize results
    results = pd.DataFrame()

    results['date_[utc]']  = pd.to_datetime(np.arange(startDate, stopDate, 
                                                        dt.timedelta(seconds = cadence)))

    results['date_[unix]']  = ((results['date_[utc]'] - pd.Timestamp("1970-01-01")) //
                                pd.Timedelta('1s'))
    
    results['mjd'] = Time(results['date_[utc]']).mjd

    results['gap']      = np.nan
    
    sample=True
    if sample:
        results_samples = [results.copy(deep=True) for _ in range(20)]
    
    # if getOrb:
        
    #     print('Generating MAVEN orbit information.')

    #     results['orb']          = np.nan

    #     orbs, apoDates = getOrbitalData()

    #     orbStart = orbs[(apoDates >= results['date_[utc]'][0])][0] - 1 
    #     orbStop  = orbs[(apoDates <  results['date_[utc]'][len(results) - 1])][-1] 


    #     for orb in np.arange(orbStart, orbStop + 1):


    #         index = (orbs == orb)

    #         startApo = apoDates[orbs == orb][0] 
    #         endApo   = apoDates[orbs == orb + 1][0]

    #         orbIndex = ((results['date_[utc]'] >= startApo) & (results['date_[utc]'] < endApo))

    #         results.loc[orbIndex, 'orb'] = orb

    print('Running from {} to {}, in {} segments, and for parameters:'.format(startDate, 
                                                                              stopDate, 
                                                                              indexStop - indexStart))
    
    for p in params:

        print('{}'.format(p))

        results['mu_{}'.format(p)]            = np.nan

        results['sigma_{}'.format(p)]         = np.nan

        # results['mu_{}_normed'.format(p)]     = np.nan

        # results['sigma_{}_normed'.format(p)]  = np.nan
    
    # arrEnum  = np.arange(indexStart, indexStop, 1)
    
    for i, o in enumerate(tqdm.tqdm(range(indexStart, indexStop+1), desc='Processing segments', position=0)):
        
        planar_data = all_planar_df.query("SubsetIndex == @o").reset_index(drop=True)
        
        for p in tqdm.tqdm(params, desc='Processing [{}]'.format(', '.join(params)), leave=False, position=1):
        
            # Add the last previous measurement and the next future measurement
            # This ensures that, at a minimum, each segment has 2 data points to connect
            last_index = all_planar_df.query("SubsetIndex == @o-1")[p].dropna().index[-1]
            next_index = all_planar_df.query("SubsetIndex == @o+1")[p].dropna().index[0]
            padded_data = pd.concat(
                [all_planar_df.query("SubsetIndex == @o-1").loc[[last_index],:], 
                 planar_data, 
                 all_planar_df.query("SubsetIndex == @o+1").loc[[next_index],:]]
                ).reset_index(drop=True)
            
            # # Backmap resulting data
            # delta_mu, delta_sigma = backmap.backmap(padded_data, x_plane=3*3396.19)
            # planar_data = backmap.make_planar(padded_data, delta_mu, delta_sigma)
            planar_data = padded_data
            
            # Results will overwrite last_index
            indResults = ((results['date_[utc]'] >= planar_data.datetime_mu.values.min()) & 
                            (results['date_[utc]'] < planar_data.datetime_mu.values.max()))
            
            # 
            # yz_res_eff = 1_000 # km
            y_res_eff = z_res_eff = yz_res_eff
            t_res_eff = cadence / (60 * 60 * 24) # seconds -> days
           
            # Cluster
            cluster_labels = cluster_by_resolution(planar_data[['y_mu', 'z_mu', 'mjd_mu']].to_numpy(), [y_res_eff, z_res_eff, t_res_eff])
            unique_cluster_labels = np.unique(cluster_labels)
            # breakpoint()
            X_mu = np.array([planar_data.iloc[cluster_labels == cl][['y_mu', 'z_mu', 'mjd_mu']].mean(axis=0) for cl in unique_cluster_labels])
            X_var = np.array(
                [np.mean(planar_data.iloc[cluster_labels == cl][['y_sigma', 'z_sigma', 'mjd_sigma']].to_numpy(), axis=0)**2 +\
                 np.std(planar_data.iloc[cluster_labels == cl][['y_mu', 'z_mu', 'mjd_mu']].to_numpy(), axis=0)**2 
                 for cl in unique_cluster_labels])
            
            # Sizes of gaps, in days
            # !!!! Need to think about how this should be calculated with multiple assets
            dist_matrix = cdist(
                Time(results.loc[indResults, 'date_[utc]']).mjd[:,None], 
                X_mu[:,2][:,None])
            results.loc[indResults, 'gap'] = np.around(np.min(dist_matrix, 1), decimals = 3)
            
            # 
            y_mu = np.array([np.nanmean(planar_data.iloc[cluster_labels == cl]['{}'.format(p)], axis=0) for cl in unique_cluster_labels])[:,None]
            y_var = np.array(
                [np.nanmean(planar_data.iloc[cluster_labels == cl]['{}_sigma'.format(p)], axis=0)**2 +\
                 np.nanstd(planar_data.iloc[cluster_labels == cl]['{}'.format(p)], axis=0)**2 
                 for cl in unique_cluster_labels])[:,None]
            
            # Sort
            ordered = X_mu[:,2].argsort()
            X_mu_ordered, X_var_ordered = X_mu[ordered,:], X_var[ordered,:]
            y_mu_ordered, y_var_ordered = y_mu[ordered,:], y_var[ordered,:]
            clean = np.isfinite(y_mu_ordered).flatten() 
            X_mu_train, X_var_train = X_mu_ordered[clean,:], X_var_ordered[clean,:]
            y_mu_train, y_var_train = y_mu_ordered[clean,:], y_var_ordered[clean,:]
            
            # log-scale appropriate quantities for training
            if p in ['n_p', 'b_mag']:
                y_mu_train_log = np.log10(y_mu_train)
                y_var_train_log = y_var_train / (y_mu_train * np.log(10))
                y_mu_train, y_var_train = y_mu_train_log, y_var_train_log
            
            # Then scale the training data
            normScaler = StandardScaler()
            normScaler.fit(y_mu_train)
            
            mmScaler = MinMaxScaler(feature_range=(0, maxRescale))
            # Force spatial dimensions onto the same scale
            X_mu_train_scaler_fit = np.vstack([X_mu_train.min(axis=0), X_mu_train.max(axis=0)])
            X_mu_train_scaler_fit[0,0:2] = X_mu_train_scaler_fit[0,0:2].min()
            X_mu_train_scaler_fit[1,0:2] = X_mu_train_scaler_fit[1,0:2].max()
            mmScaler.fit(X_mu_train_scaler_fit)
            
            X_mu_train_normed = mmScaler.transform(X_mu_train)
            X_var_train_normed = np.clip(X_var_train * mmScaler.scale_**2, 1e-5, None)
            y_mu_train_normed = normScaler.transform(y_mu_train)
            y_var_train_normed = y_var_train / normScaler.scale_**2
            
            # Remove run-dependent kernel hyperparams-- should probably reflect physical scales...
            # y & z dimensions are not independent, so build a representative kernel
            yz_dists = cdist(np.sqrt(np.sum(X_mu_train_normed[:,0:2]**2, axis=1))[:,None], 
                             np.sqrt(np.sum(X_mu_train_normed[:,0:2]**2, axis=1))[:,None])
            yz_minLength, yz_midLength, yz_maxLength = np.quantile(yz_dists[yz_dists > yz_res_eff * mmScaler.scale_[0]], [min_l, mid_l, max_l])
            yz_kernel = gpflow.kernels.RationalQuadratic(variance = init_var, active_dims=[0,1])
            # if X_mu_train_normed.shape[0] >= 5:
            #     yz_kernel.lengthscales = gpflow.Parameter(
            #         yz_midLength, transform=tfp.bijectors.SoftClip( 
            #             gpflow.utilities.to_default_float(yz_minLength),
            #             gpflow.utilities.to_default_float(yz_maxLength)))
           
            t_dists = cdist(X_mu_train_normed[:,2:3]**2, 
                            X_mu_train_normed[:,2:3]**2)
            t_minLength, t_midLength, t_maxLength = np.quantile(t_dists[t_dists != 0], [min_l, mid_l, max_l])
            t_kernel = gpflow.kernels.RationalQuadratic(variance = init_var, active_dims=[2])
            # if X_mu_train_normed.shape[0] >= 5:
            #     t_kernel.lengthscales = gpflow.Parameter(
            #         t_midLength, transform=tfp.bijectors.SoftClip( 
            #             gpflow.utilities.to_default_float(t_minLength),
            #             gpflow.utilities.to_default_float(t_maxLength)))

            composite_kernel = yz_kernel * t_kernel
            # # Iteratively build kernel for each dimension
            # kernel_list = []
            # for i in range(X_normed_train.shape[1]):
            #     dists        = cdist(X_normed_train[:,i:i+1], X_normed_train[:,i:i+1])
    
            #     dists_noZeros = dists[dists != 0]
                
            #     minLength = np.quantile(dists_noZeros, min_l)
    
            #     midLength = np.quantile(dists_noZeros, mid_l)
    
            #     maxLength = np.quantile(dists_noZeros, max_l)
    
    
            #     signal_kernel = gpflow.kernels.RationalQuadratic(variance = init_var, active_dims=[i])
    
    
            #     signal_kernel.lengthscales = gpflow.Parameter(midLength, 
            #                             transform=tfp.bijectors.SoftClip(
            #                                 gpflow.utilities.to_default_float(minLength),
            #                                 gpflow.utilities.to_default_float(maxLength)))
                
            #     kernel_list.append(signal_kernel)
            
            # composite_kernel = kernel_list[0]
            # for k in kernel_list[1:]: composite_kernel += k
            
            option = 'test'
            # breakpoint() 
            if option == 'GPR':
                model = gpflow.models.GPR((X_mu_train_normed, y_mu_trains_normed), kernel=composite_kernel)
            elif option == 'SGPR':
                rng = np.random.default_rng()
                if X_normed_train.shape[0] > 300:
                    model = gpflow.models.SGPR(
                        (X_normed_train, y_normed_train),
                        kernel=composite_kernel,
                        inducing_variable=rng.choice(X_normed_train, size=500, replace=False)
                        )
                else:
                    model = gpflow.models.GPR(
                        (X_normed_train, y_normed_train), 
                        kernel=composite_kernel
                        )
            elif option == 'GPLVM':
                rng = np.random.default_rng()
                model = gpflow.models.BayesianGPLVM(
                    data = y_normed_train[0:1000,:], 
                    X_data_mean = tf.convert_to_tensor(X_normed_train[0:1000,:]),
                    X_data_var = tf.convert_to_tensor(X_var_normed_train[0:1000,:]), 
                    kernel=gpflow.kernels.RationalQuadratic(active_dims=[0, 1, 2]), 
                    inducing_variable=rng.choice(X_normed_train, size=100, replace=False),
                    # num_inducing_variables=100,
                    X_prior_mean = X_normed_train[0:1000,:],
                    X_prior_var = X_var_normed_train[0:1000,:],
                    )
                model.likelihood.variance.assign(0.01)
                
            elif option == 'test':
                # # gpflow.config.set_default_jitter(1e-5)
                # model = gpflow.models.GPR(
                #     data=(X_mu_train_normed, y_mu_train_normed),
                #     kernel=composite_kernel,
                #     likelihood=gpflow.likelihoods.Gaussian(variance=FixedVarianceOfMean(y_var_train_normed)),
                #     )
                
                # gpflow.config.set_default_jitter(1e-5)
                model = KnownNoiseGPR(data=(X_mu_train_normed, y_mu_train_normed), 
                                      kernel=composite_kernel, 
                                      noise_var=y_var_train_normed)
                
                opt = gpflow.optimizers.Scipy() 
                
                for it in range(10):
                    pbar = tqdm.tqdm(desc='Optimizing Noisy Model iter {}'.format(it), leave=False)
                    def tqdm_callback(j):
                        pbar.update(1)
                    status = opt.minimize(model.training_loss, 
                                          model.trainable_variables,
                                          callback=tqdm_callback,
                                          method="L-BFGS-B",
                                          options={"maxiter": 200})

                    extra = input_noise_term(model, X_mu_train_normed, X_var_train_normed)
                    new_noise = y_var_train_normed.reshape(-1) + extra
                    old_noise = model.known_noise.numpy().copy()
                    model.known_noise.assign(new_noise)
                    
                    rel = np.max(np.abs(new_noise - old_noise)) / np.max(new_noise)
                    if rel < 1e-2:
                        break
                if np.isnan(rel):
                    print("Noisy Model Failed to Optimize")
                    breakpoint()
                else:
                    print("Noisy Model Optimized with rel = {}".format(rel))
                    
            #----------and now save results
            t_model = results.loc[indResults, 'mjd'].to_numpy().reshape(-1,1)
            X_model = mmScaler.transform(np.hstack([t_model*0, t_model*0, t_model]))

            # Replace sampling with embarassingly parallel sampling
            def predict_y(X_):
                n_cpus = multiprocessing.cpu_count() - 1
                X_model_segments = np.array_split(X_, n_cpus)
                pred_y = Parallel(n_jobs=n_cpus)(delayed(model.predict_y)(_X) for _X in X_model_segments)
                
                mean_model = np.concatenate([py[0] for py in pred_y])
                var_model = np.concatenate([py[1] for py in pred_y])
    
                std_model = np.sqrt(var_model)
    
                # results.loc[indResults, 'mu_{}_normed'.format(p)]    = mean_model
    
                # results.loc[indResults, 'sigma_{}_normed'.format(p)] = std_model
    
                mean_model_unnorm = normScaler.inverse_transform(mean_model.reshape(-1, 1))[:, 0]
    
                std_model_unnorm  = (std_model*normScaler.scale_).flatten()
                
                return mean_model_unnorm, std_model_unnorm
            
            def predict_f(X_):
                n_cpus = multiprocessing.cpu_count() - 1
                X_model_segments = np.array_split(X_, n_cpus)
                pred_f = Parallel(n_jobs=n_cpus)(delayed(model.predict_f)(_X) for _X in X_model_segments)
                
                mean_model = np.concatenate([pf[0] for pf in pred_f])
                var_model = np.concatenate([pf[1] for pf in pred_f])
    
                std_model = np.sqrt(var_model)
    
                # results.loc[indResults, 'mu_{}_normed'.format(p)]    = mean_model
    
                # results.loc[indResults, 'sigma_{}_normed'.format(p)] = std_model
    
                mean_model_unnorm = normScaler.inverse_transform(mean_model.reshape(-1, 1))[:, 0]
    
                std_model_unnorm  = (std_model*normScaler.scale_).flatten()
                
                return mean_model_unnorm, std_model_unnorm
            
            def predict_f_samples(X_, num_samples=1):
                n_cpus = multiprocessing.cpu_count() - 1
                X_model_segments = np.array_split(X_, n_cpus)
                samples = Parallel(n_jobs=n_cpus)(delayed(model.predict_f_samples)(_X, num_samples) for _X in X_model_segments)
                
                combined_samples = np.hstack(samples)
    
                sample_unnorm = normScaler.inverse_transform(combined_samples.reshape(-1, 1)).reshape(num_samples, -1, 1)
                
                return sample_unnorm
            
            mean_model_unnorm, std_model_unnorm = predict_f(X_model)     
            
            if sample:
                sample_unnorm = predict_f_samples(X_model, len(results_samples))
            
            if p in ['n_p', 'b_mag']:
                mean_model_unnorm = 10**mean_model_unnorm
                std_model_unnorm = mean_model_unnorm * np.log(10) * std_model_unnorm
                
                if sample:
                    sample_unnorm = 10**sample_unnorm
            
            results.loc[indResults, 'mu_{}'.format(p)]    = mean_model_unnorm

            results.loc[indResults, 'sigma_{}'.format(p)] = std_model_unnorm
            
            if sample: 
                for _ in range(len(results_samples)):
                    results_samples['{}_sample'.format(_)].loc[indResults, p] = sample_unnorm[_]
            
            asset_colors = {"MAVEN": "xkcd:rust", "MEX": "xkcd:red orange", "Tianwen-1": "xkcd:goldenrod"}
            
            # Quick plot 
            fig, ax = plt.subplots()
            
            for category, group in planar_data.groupby("asset"):
                ax.scatter(group['mjd_mu'], group[p], color=asset_colors[category], marker='.', s=8, lw=0, label=category, zorder=10)
            ax.scatter(X_mu[:,2], y_mu, color='blue', marker='.', s=16, lw=0, zorder=12)
            ax.errorbar(X_mu[:,2], y_mu.flatten(), yerr=np.sqrt(y_var.flatten()), color='blue', ls='None', elinewidth=1, capsize=0)
            ax.plot(t_model, mean_model_unnorm, lw=1, color='xkcd:purple')
            ax.fill_between(t_model.flatten(), mean_model_unnorm-std_model_unnorm, mean_model_unnorm+std_model_unnorm, 
                            fc=("xkcd:lavender", 0.5), ec=("xkcd:purple", 0.75), zorder=11)
            ax.legend()
            plt.show()
            
            
            # test = np.array(model.predict_f_samples(X_model, 10))
            # for i in range(len(test)):
            #     test[i] = normScaler.inverse_transform(test[i])
                
            # # Test sampling elsewhere -- e.g. 3 R_M away?
            # X_model_p2 = mmScaler.transform(np.hstack([t_model*0+6792.38, t_model*0, t_model]))
            # mean_model_unnorm_p2, std_model_unnorm_p2 = predict_y(X_model_p2) 
            
            # X_model_m2 = mmScaler.transform(np.hstack([t_model*0-6792.38, t_model*0, t_model]))
            # mean_model_unnorm_m2, std_model_unnorm_m2 = predict_y(X_model_m2)
            
            # # fig, ax = plt.subplots(figsize=(8,2.5))
            # # plt.subplots_adjust(left=0.1, bottom=0.2, top=0.975, right=0.975)
            # fig, ax = plt.subplots(figsize=(8,2))
            # plt.subplots_adjust(left=0.1, bottom=0.05, top=0.975, right=0.975)
            
            # ax.plot(results['date_[utc]'], np.abs(mean_model_unnorm), color='black', lw=2, zorder=10)
            # ax.fill_between(results['date_[utc]'], np.abs(mean_model_unnorm)+std_model_unnorm, np.abs(mean_model_unnorm)-std_model_unnorm,
            #                 fc=('black', 0.1), ec=('black', 0.5), lw=1, zorder=10)
            
            # ax.plot(results['date_[utc]'], np.abs(test[0].flatten()), color='black', lw=1, zorder=1)
            # ax.plot(results['date_[utc]'], np.abs(test[1].flatten()), color='black', lw=1, zorder=1)
            # ax.plot(results['date_[utc]'], np.abs(test[2].flatten()), color='black', lw=1, zorder=1)
            # ax.plot(results['date_[utc]'], np.abs(test[3].flatten()), color='black', lw=1, zorder=1)
            
            # # ax.plot(results['date_[utc]'], mean_model_unnorm_p2, color='red', lw=1)
            # # ax.fill_between(results['date_[utc]'], mean_model_unnorm_p2+std_model_unnorm_p2, mean_model_unnorm_p2-std_model_unnorm_p2,
            # #                 fc=('red', 0.1), ec=('red', 0.5), lw=1)
            
            # # ax.plot(results['date_[utc]'], mean_model_unnorm_m2, color='xkcd:dark orange', lw=1)
            # # ax.fill_between(results['date_[utc]'], mean_model_unnorm_m2+std_model_unnorm_m2, mean_model_unnorm_m2-std_model_unnorm_m2,
            # #                 fc=('xkcd: dark orange', 0.1), ec=('xkcd:dark orange', 0.5), lw=1)
            
            # ax.scatter(data.query("asset == 'MEX'")['datetime_mu'], data.query("asset == 'MEX'")[p], 
            #            color='xkcd:cyan', marker='.', s=40, lw=0, zorder=20)
            # ax.scatter(data.query("asset == 'MAVEN'")['datetime_mu'], data.query("asset == 'MAVEN'")[p], 
            #            color='xkcd:lilac', marker='.', s=40, lw=0, zorder=20)
            
            # ax.set(xlim=[results['date_[utc]'].min(), results['date_[utc]'].max()], 
            #        ylim=[350, 550], ylabel = 'Solar Wind Speed [km/s]')
            #        # ylim=[0, 15], ylabel = r'Proton Density [$cm^{-3}$]')
            #        # ylim=[0, 12], ylabel = r'IMF Magnitude [$nT$]', xlabel='Date')
            # plt.show()
            
            
            # fig, ax = plt.subplots()
            # ax.plot(results['mjd'], results['mu_v_mag'], lw=1.5, color='xkcd:cornflower')
            # ax.fill_between(results['mjd'], 
            #                 results['mu_v_mag']-results['sigma_v_mag'], 
            #                 results['mu_v_mag']+results['sigma_v_mag'], 
            #                 fc=('xkcd:cornflower', 0.5), ec=('xkcd:cornflower', 0.75))

            # ax.errorbar(data['mjd_mu'], data['v_mag'], xerr=data['mjd_sigma'], yerr=data['v_mag_sigma'], color='black', ls='None', elinewidth=0.5)

            # # ax.set(xlim=[results['mjd'].min(), results['mjd'].max()], xlabel='MJD', ylabel='Solar Wind Speed [km/s]')
            # ax.set(xlabel='MJD', ylabel='Solar Wind Speed [km/s]')
            # plt.show()
            # breakpoint()
            # # Sample on grid
            # y_grid = np.linspace(-20_000, 20_000, 200)
            # z_grid = np.linspace(-10_000, 10_000, 100)
            # t_grid = np.unique(np.floor(t_model))
            
            # yg, zg, tg = np.meshgrid(y_grid, z_grid, t_grid, indexing='ij')
            # X_grid = mmScaler.transform(np.hstack([yg.flatten()[:,None], zg.flatten()[:,None], tg.flatten()[:,None]]))
            
            # mean_g_normed, var_g_normed = model.predict_y(X_grid)
            
            # mean_g = normScaler.inverse_transform(mean_g_normed)
            # mean_g = mean_g.flatten().reshape(yg.shape)
            # var_g = np.sqrt(var_g_normed) * normScaler.scale_
            # var_g = var_g.flatten().reshape(yg.shape)
            
            # fig, axs = plt.subplots(nrows=int(np.floor(np.sqrt(len(t_grid)))), 
            #                         ncols=int(len(t_grid)/np.floor(np.sqrt(len(t_grid)))), 
            #                         figsize=(8,8), sharex=True, sharey=True)
            # axs = axs.flatten()
            # for iax, ax in enumerate(axs):
            #     ax.pcolormesh(yg[:,:,iax], zg[:,:,iax], mean_g[:,:,iax], 
            #                   vmin=300, vmax=600, cmap='plasma')
            #     ax.scatter(data.query("@t_grid[@iax] <= mjd_mu < @t_grid[@iax]+1")['y_mu'], 
            #                data.query("@t_grid[@iax] <= mjd_mu < @t_grid[@iax]+1")['z_mu'], 
            #                marker='o', s=10, color='black')
                
            #     ax.set(xlim=[yg.min(), yg.max()], ylim=[zg.min(), zg.max()], aspect=1)
                
            #     ax.annotate("{:.0f}".format(t_grid[iax]), (0,1), (1,-1), 
            #                 xycoords='axes fraction', textcoords='offset fontsize', 
            #                 ha='left', va='top')
            
        # breakpoint()
        # # Test the model predictions against data
        # test_points = data[['y_mu', 'z_mu', 'mjd_mu']].to_numpy()
        # test_points_normed = mmScaler.transform(test_points)
        # test_y_mu_normed, test_y_var_normed = model.predict_y(test_points_normed)
        # test_y_mu = normScaler.inverse_transform(test_y_mu_normed)
        # test_y_sigma = np.sqrt(test_y_var_normed) * normScaler.scale_
        
        # fig, ax = plt.subplots()
        # ax.scatter(data['mjd_mu'], data['{}'.format(p)], color='black', marker='o', s=10)
        
        # ax.errorbar(test_points[:,2], test_y_mu[:,0], yerr=test_y_sigma[:,0], color='red', ls='None', elinewidth=0.5, capsize=0.1)
        # ax.set(ylim=[200, 700])
        
        # test_points[:,1] += 20_000
        # test_points_normed = mmScaler.transform(test_points)
        # test_y_mu_normed, test_y_var_normed = model.predict_y(test_points_normed)
        # test_y_mu = normScaler.inverse_transform(test_y_mu_normed)
        # test_y_sigma = np.sqrt(test_y_var_normed) * normScaler.scale_
        
        # fig, ax = plt.subplots()
        # ax.scatter(data['mjd_mu'], data['{}'.format(p)], color='black', marker='o', s=10)
        
        # ax.errorbar(test_points[:,2], test_y_mu[:,0], yerr=test_y_sigma[:,0], color='red', ls='None', elinewidth=0.5, capsize=0.1)
        # ax.set(ylim=[200, 700])
        # plt.show()
        # diffs = test_y_mu[:,0] - data[p].to_numpy()
        # diffs_scaled = diffs / test_y_sigma[:,0]
    
    breakpoint()
    
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
