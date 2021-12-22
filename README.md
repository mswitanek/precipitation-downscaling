# precipitation-downscaling

The scripts "downscalingPrecipitation_ForcedByOBS.py" and "downscalingPrecipitation_ForcedByRCM.py" implement a stochastic downscaling model that uses gridded precipitation to produce spatially coherent sub-grid precipitation fields using a transformed Gaussian model. The files "precipDataOBS.json.gz" and "precipDataRCM.json.gz" are gzipped json files containing the observed and bias corrected RCM precipitation values for eight catchments across Austria. The data files contain gridded predictor fields along with station data for these catchments and the 12 calendar months. This downscaling model produces simulations, at the station-level, given time series of gridded precipitation. 

The scripts are designed to be run interactively. For example, after starting Python, run from your prompt:

">>>exec(open("downscalingPrecipitation_ForcedByOBS.py",'r').read())"

or from an IPython prompt:

">>>%run -i downscalingPrecipitation_ForcedByOBS.py"

The code and data herein are used to provide an example implementation of the stochastic downscaling model. Ultimately, the user will need to adapt the code to their own case. 
