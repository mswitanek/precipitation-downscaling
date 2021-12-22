# -*- coding: utf-8 -*-

################################################################################
__help__ = """
    Code to provide stochastically downscaled precipitation 
    simulations with a coherent spatial structure. This script downscales 
    gridded observational precipitation data to station data. The script can be 
    run from the Python prompt as:
    >>>exec(open("downscalingPrecipitation_ForcedByOBS.py",'r').read())
    or from IPython as:
    >>>%run -i downscalingPrecipitation_ForcedByOBS.py
    """
__author__ = "Matt Switanek"
__copyright__ = "Copyright 2021, Matt Switanek" 
################################################################################

import os
import gzip
import json
import time
import datetime 
import numpy as np

################################################################################
##### Load in data from gzipped json file 
################################################################################

### The names of the catchments used in this study
catchmentNames = ["ObereMur","IllSugadin","Pitztal","ObereIsel"
    ,"Defreggental","Gailtal","Pitten","Palten"]
### Months of the year
months = ["January","February","March","April","May","June","July"
    ,"August","September","October","November","December"]

f1 = gzip.open("precipDataOBS.json.gz",'r')
dataObs = json.load(f1)
f1.close()

### dataObs is a dictionary that contains the observed data
for catchment in catchmentNames:
    for month in months:
        dataObs[catchment][month]['latLonPredictors']\
            = np.array(dataObs[catchment][month]['latLonPredictors'])
        dataObs[catchment][month]['latLonPredictands']\
            = np.array(dataObs[catchment][month]['latLonPredictands'])
        dataObs[catchment][month]['precipitation']\
            = np.array(dataObs[catchment][month]['precipitation'])/100.

###############################################################################################################
##### Implement the downscaling model using observed gridded values at ~12km 
#####       resolution as the predictors and the station point data are the predictands
###############################################################################################################

###This is the number applied to a save folder containing the different simulations.
###So, for example a value of "1" would contain one set of all 50 simulations.
obsSimNumber = "1"

### Loop through the 8 catchments for this study
for catchment in catchmentNames[0:8]:
    ### Loop through the months of the year 
    for month in months[0:12]:
        ### Loop through the different simulations
        for simnum in range(0,50):

            ### Data contains the daily precipitation amounts for or predictors 
            ###     and our predictands
            data = np.array(dataObs[catchment][month]['precipitation'].T)
            numberOfPredictors = np.array(dataObs[catchment][month]['numberOfPredictors'])
            numberOfPredictands = np.array(dataObs[catchment][month]['numberOfPredictands'])
            latlonPredictors = np.array(dataObs[catchment][month]['latLonPredictors'])
            latlonPredictands = np.array(dataObs[catchment][month]['latLonPredictands'])
            
            ##### Get the area averaged precipitaiton amount of the predictor field. 
            ##### We will only simulate predictand values other than zero, when our field is not zero.
            areaAvgPrecip = np.mean(data[:numberOfPredictors,:],axis=0)
            
            ### Get two randomly generated periods of time, where one is used for
            ###     for calibration and the other is used for validation
            ### Or, one could split the time and use the first half to calibrate 
            ###     and the second half to validate
            halflen = int(data.shape[1]/2)
            total_days = np.arange(0,data.shape[1])
            days2fit = np.argsort(np.random.rand(data.shape[1]))[:halflen]
            days2predict = np.delete(total_days,days2fit)

            ### Transform the data by the third root
            transform = 3.
            dataT = np.array(data**(1/transform))

            ### Calculate the covariance matrix (over the calibration period) of the 
            ###     observed data containing both the predictors and predictands 
            cv1 = np.cov(dataT[:,days2fit])                       
            try:
                ##### If covariance matrix is positive definite
                cholesky1 = np.linalg.cholesky(cv1)
            except:
                ##### If covariance matrix is not positive definite
                positiveDefinite = False
                cv1alt = np.array(cv1)
                while positiveDefinite == False:
                    u,s,v = np.linalg.svd(cv1alt)
                    s[s<1e-12] = 1e-12
                    dd = np.diag(s)
                    cv1alt = np.dot(np.dot(u,dd),u.T)
                    try:
                        cholesky1 = np.real(np.linalg.cholesky(cv1alt))
                        positiveDefinite = True
                    except:
                        positiveDefinite = False
                

            ### Randomly generate an initial transformed simulation, given the transformed predictor field
            ### The arrays sim1, sim2, sim3, sim4 are all different steps to produce the final simulation
            ### sim1 produces the initial randomized, Gaussian simulation
            ### sim2 retransforms sim1
            ### sim3 removes quantile-to-quantile biases of the simulation
            ### sim4 recorrelates our entire cross-correlation matrix including the predictors and predictands
            sim1 = np.zeros((dataT.T.shape))
            sim1[:,:numberOfPredictors] = np.array(dataT[:numberOfPredictors,:].T)
            conditionalRandom = np.zeros((dataT.shape[0]))
            for v1 in range(0,dataT.shape[1]): 
                pcurr = dataT[:numberOfPredictors,v1+0]
                if areaAvgPrecip[v1+0] > 0.0:
                    ### Option 1: This option is provided for illustrative purposes 
                    ### Comment out the next five lines if option 2 is uncommented 
                    #####for v2 in range(0,numPredictors):
                        #####if v2 == 0:
                            #####conditionalRandom[v2] = pcurr[v2]/cholesky1[v2,v2]
                        #####else:
                            #####conditionalRandom[v2] = (pcurr[v2]-np.sum(conditionalRandom[:v2]*cholesky1[v2,:v2]))/cholesky1[v2,v2]                 
                    ### Option 2: This produces the same results as Option 1, though it will run faster.
                    conditionalRandom[:numberOfPredictors] = \
                        np.linalg.lstsq(cholesky1[:numberOfPredictors,:numberOfPredictors]\
                        ,pcurr,rcond=None)[0]
                    conditionalRandom[numberOfPredictors:] = np.random.normal(0,1.0,size=numberOfPredictands)
                    sim1[v1,numberOfPredictors:] = \
                        np.dot(cholesky1[numberOfPredictors:,:],conditionalRandom)
            sim1[sim1<0] = 0
            
            ### sim2 retransforms sim1
            sim2 = np.array(sim1**transform)
           
            ### sim3 removes quantile-to-quantile biases of the simulation
            ### Use the calibration period to bias correct (multiplicatively) 
            ###     the different quantiles of the distribution for the entire period
            sim3 = np.zeros(sim2.shape)
            sim3[:,:numberOfPredictors] = np.array(data[:numberOfPredictors,:].T)
            for pnum in range(0,numberOfPredictands):                
                offset1 = np.sort(data[numberOfPredictors+pnum,days2fit]+0.1)\
                    /np.sort(sim2[days2fit,numberOfPredictors+pnum]+0.1)
                ### Set maximum and minimum thresholds for our observed/simulated quantile ratios.
                ### This is used to avoid any potential simulated values in the distribution that 
                ###     could potentially be dramatically different from observed values.
                offset1[offset1>5] = 5.
                offset1[offset1<0.2] = 0.2
                offset = np.interp(np.linspace(0,1,data.shape[1]),np.linspace(0,1.0,days2fit.shape[0]),offset1)
                srtvals = np.argsort(sim2[:,numberOfPredictors+pnum])
                sim3[srtvals,numberOfPredictors+pnum] = sim2[srtvals,numberOfPredictors+pnum]*offset

            ### sim4 recorrelates our entire cross-correlation matrix including the predictors and predictands
            ### Recorrelation 
            cor_obs = np.corrcoef(data[:,days2fit])
            cor_mod = np.corrcoef(sim3[:,:].T)
            u,s,v = np.linalg.svd(cor_obs)
            s1 = np.zeros(cor_obs.shape)
            s1[:] = np.diag(s)
            S = np.dot(u, np.dot(s1**.5, u.T))                
            uu,ss,vv = np.linalg.svd(cor_mod)
            s2 = np.zeros(cor_mod.shape)
            s2[:] = np.diag(ss**(-.5))
            T = np.dot(uu, np.dot(s2, uu.T))
            F = np.dot(T,S)
            sim4 = np.dot(sim3,F)
            sim4[sim4<0.0] = 0
            

            ##### Use the calibration period to correct any mean bias of the model
            for pnum in range(0,numberOfPredictands):                
                meandiff = np.mean(data[numberOfPredictors+pnum,days2fit]) \
                    / np.mean(sim4[days2fit,numberOfPredictors+pnum])                     
                sim4[:,numberOfPredictors+pnum] = sim4[:,numberOfPredictors+pnum]*meandiff

            sim4[:,:numberOfPredictors] = np.array(sim3[:,:numberOfPredictors]) 
            ##### Set any values less than 0.1 mm to 0.0 mm            
            sim4[sim4<0.1] = 0
            simulated_vals = np.array(sim4)


            ##### Save simulations as a numpy files
            if os.path.exists("simulations") == False:
                os.mkdir("simulations")

            if os.path.exists("simulations/conditionalOBS"+obsSimNumber) == False:
                os.mkdir("simulations/conditionalOBS"+obsSimNumber)

            if os.path.exists("simulations/conditionalOBS"+obsSimNumber+"/"+catchment) == False:
                os.mkdir("simulations/conditionalOBS"+obsSimNumber+"/"+catchment)
              
            if simnum == 0:
                np.savez("simulations/conditionalOBS"+obsSimNumber+"/"+catchment\
                    +"/"+catchment+"_"+month+"_Sim"+str(simnum+1)\
                    ,observed_data=np.float32(data.T),simulated_data=np.float32(simulated_vals)\
                    ,latlonPredictors=np.float32(latlonPredictors)\
                    ,latlonPredictands=np.float32(latlonPredictands)\
                    ,days2fit=days2fit,days2predict=days2predict)
            else:
                np.savez("simulations/conditionalOBS"+obsSimNumber+"/"+catchment\
                    +"/"+catchment+"_"+month+"_Sim"+str(simnum+1)\
                    ,simulated_data=np.float32(simulated_vals),days2fit=days2fit,days2predict=days2predict)

        print(catchment,month)

 
 
 
