# -*- coding: utf-8 -*-

################################################################################
__help__ = """
    Code to provide stochastically downscaled precipitation 
    simulations with a coherent spatial structure. This script downscales 
    gridded RCM precipitation data to station data. The script can be 
    run from the Python prompt as:
    >>>exec(open("downscalingPrecipitation_ForcedByRCM.py",'r').read())
    or from IPython as:
    >>>%run -i downscalingPrecipitation_ForcedByRCM.py
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
##### Load in data from gzipped json files
################################################################################

### The names of the catchments used in our example study
catchmentNames = ["ObereMur","IllSugadin","Pitztal","ObereIsel"
    ,"Defreggental","Gailtal","Pitten","Palten"]
### Months of the year
months = ["January","February","March","April","May","June","July"
    ,"August","September","October","November","December"]

f1=gzip.open("precipDataOBS.json.gz",'r')
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

f2=gzip.open("precipDataRCM.json.gz",'r')
dataMod = json.load(f2)
f2.close()

### dataMod is a dictionary that contains the RCM data
for catchment in catchmentNames:
    for month in months:
        dataMod[catchment][month]['latLonPredictors']\
            = np.array(dataMod[catchment][month]['latLonPredictors'])
        dataMod[catchment][month]['precipitation']\
            = np.array(dataMod[catchment][month]['precipitation'])/100.

##### Time series of the current RCM model data 
endDate = datetime.date(2100,1,1).toordinal()
startDate = datetime.date(1950,1,1).toordinal()
mod_timeseries = np.zeros((endDate-startDate,3),dtype=np.uint16)
for dts in range(0,len(mod_timeseries[:,0])):
    dateNow = datetime.date.fromordinal(dts+startDate)
    mod_timeseries[dts,0] = dateNow.year
    mod_timeseries[dts,1] = dateNow.month
    mod_timeseries[dts,2] = dateNow.day 

###############################################################################################################
##### Implement the downscaling model using observed gridded values at ~12km 
#####       resolution as the predictors and the station point data are the predictands
###############################################################################################################

###This is the number applied to a save folder containing the different simulations.
###So, for example a value of "1" would contain one set of all 50 simulations.
modSimNumber = "1"

### Loop through the 8 catchments for this study
for catchment in catchmentNames[0:8]:
    ### Loop through the months of the year 
    for month in months[0:12]:
        ### Get the subset of model time overlapping with observed record. This is used to correct for model bias.
        modSubDates1 = np.nonzero(mod_timeseries[:,1]==months.index(month)+1)[0]
        modSubDates2 = np.nonzero(mod_timeseries[modSubDates1,0]>1960)[0]
        modSubDates3 = np.nonzero(mod_timeseries[modSubDates1[modSubDates2],0]<2011)[0]
        modSubDates = modSubDates2[modSubDates3]
 
        ### Loop through the different simulations
        for simnum in range(0,50):

            ### The observational data for the current catchment and month
            ### The "data" array contains the oberved data, while the RCM data 
            ###     is contained in the dataModPredictors array (see below)
            data = np.array(dataObs[catchment][month]['precipitation']).T
            numberOfPredictors = np.array(dataObs[catchment][month]['numberOfPredictors'])
            numberOfPredictands = np.array(dataObs[catchment][month]['numberOfPredictands'])
            latlonPredictors = np.array(dataObs[catchment][month]['latLonPredictors'])
            latlonPredictands = np.array(dataObs[catchment][month]['latLonPredictands'])
            
            ### The RCM model data for the current catchment and month
            dataModPredictors = np.array(dataMod[catchment][month]['precipitation'].T)

            ### The days from the observed record that we are using to fit the model
            days2fit = np.arange(0,data.shape[1])
                        
            ### Recorrelate the RCM correlation matrix to that of the observed 
            ###     correlation matrix. This is done for the predictors only.
            if simnum == 0:
                cor_obs = np.corrcoef(data[:numberOfPredictors,days2fit])
                cor_mod = np.corrcoef(dataModPredictors[:,:])
                u,s,v = np.linalg.svd(cor_obs)
                s1 = np.zeros(cor_obs.shape)
                s1[:] = np.diag(s)
                S = np.dot(u, np.dot(s1**.5, u.T))                
                uu,ss,vv = np.linalg.svd(cor_mod)
                s2 = np.zeros(cor_mod.shape)
                s2[:] = np.diag(ss**(-.5))
                T = np.dot(uu, np.dot(s2, uu.T))
                F = np.dot(T,S)
                dataModPredictorsRecorr = np.dot(dataModPredictors.T,F).T
                dataModPredictorsRecorr[dataModPredictorsRecorr<0] = 0
                
            ### Apply the transformation to the observed and the recorrelated RCM precipitation data
            transform = 3.
            dataT = np.array(data**(1/transform))
            dataModPredictorsTRecorr = np.array(dataModPredictorsRecorr**(1/transform))

            ### Calculate the observed covariance matrix, for the predictors and 
            ###     the predictands, and use this to find the Cholesky decomposition
            cv1 = np.cov(dataT[:,days2fit])                       
            try:
                ##### If covariance matrix is positive definite
                cholesky1 = np.linalg.cholesky(cv1)
            except:
                ##### If covariance matrix is not positive definite, then fix that
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

            ### Get the area averaged precipitation amount of the predictor field. 
            ###       We will only simulate predictand values other than zero, when our field is not zero.
            areaAvgPrecip = np.mean(dataModPredictorsTRecorr[:numberOfPredictors,:],axis=0)

            ### Randomly generate an initial transformed simulation, given the transformed predictor field
            ### The arrays sim1, sim2, sim3, sim4 are all different steps to produce the final simulation
            ### sim1 produces the initial randomized, Gaussian simulation
            ### sim2 retransforms sim1
            ### sim3 removes quantile-to-quantile biases of the simulation
            ### sim4 recorrelates our entire cross-correlation matrix including the predictors and predictands
            sim1 = np.zeros((dataModPredictors.shape[1],data.shape[0]))
            sim1[:,:numberOfPredictors] = np.array(dataModPredictorsTRecorr[:numberOfPredictors,:].T)
            conditionalRandom = np.zeros((data.shape[0]))
            for v1 in range(0,dataModPredictors.shape[1]): 
                pcurr = dataModPredictorsTRecorr[:numberOfPredictors,v1+0]
                if areaAvgPrecip[v1+0] > 0.0:
                    #####for v2 in range(0,numberOfPredictors):
                        #####if v2 == 0:
                            #####conditionalRandom[v2] = pcurr[v2]/cholesky1[v2,v2]
                        #####else:
                            #####conditionalRandom[v2] = (pcurr[v2]-np.sum(conditionalRandom[:v2]*cholesky1[v2,:v2]))/cholesky1[v2,v2]        
                    conditionalRandom[:numberOfPredictors] = \
                        np.linalg.lstsq(cholesky1[:numberOfPredictors,:numberOfPredictors]\
                        ,pcurr,rcond=None)[0]
                    conditionalRandom[numberOfPredictors:] = np.random.normal(0,1.0,size=numberOfPredictands)
                    sim1[v1,numberOfPredictors:] = np.dot(cholesky1[numberOfPredictors:,:],conditionalRandom)
            sim1[sim1<0] = 0

            ### sim2 retransforms sim1
            sim2 = np.array(sim1**transform)

            ### sim3 removes quantile-to-quantile biases of the simulation
            sim3 = np.zeros(sim2.shape)          
            for pnum in range(0,sim2.shape[1]):                
                offset1 = np.sort(data[pnum,days2fit]+.1)\
                    /np.sort(sim2[modSubDates[0]+np.argsort(np.random.rand(modSubDates.shape[0]))[:days2fit.shape[0]],pnum]+.1)
                ### Set maximum and minimum thresholds for our observed/simulated quantile ratios.
                ### This is used to avoid any potential simulated values in the distribution that 
                ###     could potentially be dramatically different from observed values.
                offset1[offset1>5] = 5.
                offset1[offset1<0.2] = 0.2
                offset = np.interp(np.linspace(0,1,sim2.shape[0]),np.linspace(0,1.0,days2fit.shape[0]),offset1)
                srtvals = np.argsort(sim2[:,pnum])
                sim3[srtvals,pnum] = sim2[srtvals,pnum]*offset

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

            ### Lastly, correct any mean bias with respect to the observations in 
            ###     the calibration period.
            for pnum in range(0,sim4.shape[1]):                
                meandiff = np.mean(data[pnum,days2fit]) / np.mean(sim4[modSubDates,pnum])                     
                sim4[:,pnum] = sim4[:,pnum]*meandiff
            
            sim4[:,:numberOfPredictors] = np.array(dataModPredictorsRecorr[:numberOfPredictors,:].T)
            sim4[sim4<0.1] = 0
            simulated_vals = np.array(sim4)
            
            
            ### Save simulations as a numpy file
            if os.path.exists("simulations") == False:
                os.mkdir("simulations")

            if os.path.exists("simulations/conditionalRCM"+modSimNumber) == False:
                os.mkdir("simulations/conditionalRCM"+modSimNumber)

            if os.path.exists("simulations/conditionalRCM"+modSimNumber+"/"+catchment) == False:
                os.mkdir("simulations/conditionalRCM"+modSimNumber+"/"+catchment)

            if simnum == 0:
                np.savez("simulations/conditionalRCM"+modSimNumber+"/"+catchment\
                    +"/"+catchment+"_"+month+"_Sim"+str(simnum+1)\
                    ,observed_data=np.float32(data.T),simulated_data=np.float32(simulated_vals)\
                    ,mod_predictor_data=np.float32(dataModPredictors.T)\
                    ,latlonPredictors=np.float32(latlonPredictors)\
                    ,latlonPredictands=np.float32(latlonPredictands))
            else:
                np.savez("simulations/conditionalRCM"+modSimNumber+"/"+catchment\
                    +"/"+catchment+"_"+month+"_Sim"+str(simnum+1)\
                    ,simulated_data=np.float32(simulated_vals))

        print(catchment,month)
            



