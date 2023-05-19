# Wi-Fi multipath parameter estimation 

This repository contains the reference code for the article [''Wi-Fi Multi-Path Parameter Estimation for Sub-7 GHz Sensing: A Comparative Study'']().

If you find the project useful and you use this code, please cite our article:
```
@inproceedings{meneghello2023wifimultipath,
  author={Meneghello, Francesca and Blanco, Alejandro and Cusano, Antonio and Widmer, Joerg and Rossi, Michele},
  booktitle={Proc. of IEEE WiMob}, 
  title={{Wi-Fi Multi-Path Parameter Estimation for Sub-7 GHz Sensing: A Comparative Study}}, 
  year={2023},
  address={Montreal, Canada}
  }
```

## How to use

If you want to replicate the quantitative evaluation in the paper above you need to generate simulated channel frequency response data using the ```data_generation.m``` code as explained in the first step.

If you want to use the multi-path parameter estimators on your own data skip the generation of the simulated files and directly use the estimators.

## Simulated data generation
To generate the channel data for the evaluation of the multi-path parameter estimation algorithms use:
```bash 
matlab Matlab_code/data_generation.m
```

## Multi-path parameter estimation approaches
The following pipelines take as input a file containing channel frequency response (CFR) data and provide as output the estimated parameters of the multi-path components. 

### mD-Track pipeline
```bash 
python Python_code/method_mDtrack.py <'number of spatial streams'> <'number of cores'> <'name of the directory of data'> <'starting of the name of the file'> <'delta ToA for grid search in multiples of 10^-11'>
```
e.g., 
python Python_code/method_mDtrack.py 1 4 ../simulation_files/change_delay_aoa/ simulation_artificial_grid_delaydiff_1e-09_aoadiff_4 --delta_t 25

The script uses the following utility functions:
```build_aoa_matrix```, ```build_toa_matrix``` in ```utilityfunct_aoa_toa_doppler.py``` 
and ```md_track_2d``` in ```utilityfunct_md_track.py```.

### Compressed sensing pipeline (IHT w/o refinement, IHT with refinement, OMP, LASSO)
```bash 
python Python_code/method_compressed_sensing.py <'start index for processing'> <'end index for processing'> <'step length'> <'optimization method (among iht_noref, iht, omp, lasso)'> <'name of the directory'> <'starting of the name of the file'>
```
e.g., 
python Python_code/method_compressed_sensing_simulation.py 0 -1 1 iht ../simulation_files/change_delay_aoa/ simulation_artificial_grid_delaydiff_1e-09_aoadiff_4

The script uses the following utility functions:
```build_toa_matrix```, ```build_toa_aoa_matrix``` in ```utilityfunct_aoa_toa_doppler.py```,
and the functions in ```utilityfunct_optimization_routines.py``` and ```utilityfunct_optimization.py```. Note that the LASSO routine requires the [OSQP solver](https://osqp.org/).

### SpotFi pipeline
```bash 
matlab Matlab_code/method_SpotFi.m
```
The script uses the functions inside the ```functions``` folder.

### UbiLocate pipeline
```bash 
matlab Matlab_code/method_UbiLocate_2D.m
```
The script uses the functions inside the ```functions``` folder.


## Performance evaluation
The following pipelines analyze the results obtained through the multi-path parameter estimators and compute the average performance. 

### For mD-Track, IHT w/o refinement, IHT with refinement, OMP, LASSO
```bash 
python Python_code/analysis_from_python_methods.py <'step for the AoA of first path'> <'step for the AoA of second path'> <'maximum number of paths detected'> <'optimization method (among iht_noref, iht, omp, lasso)'> <'name of the directory'> <'starting of the name of the file'>
```
e.g., python Python_code/analysis_from_python_methods.py 1 2 2 mdTrack ../simulation_files/change_delay_aoa/ simulation_artificial_delayobst_1.86e-08

### For UbiLocate and SpotFi
```bash 
python Python_code/analysis_from_matlab_methods.py <'step for the AoA of first path'> <'step for the AoA of second path'> <'maximum number of paths detected'> <'optimization method (among iht_noref, iht, omp, lasso)'> <'name of the directory'> <'starting of the name of the file'> <'whether the amplitudes are in dB (default 0, i.e., not dB)'>
```
e.g., python Python_code/analysis_from_matlab_methods.py 1 2 2 ubilocate ../simulation_files/change_delay_aoa/ simulation_artificial_delayobst_1.86e-08 --dB 0

For ```spotfi``` use --dB 1.

### Plotting the results
```bash 
python Python_code/plot_combined.py <'start for ToA'> <'end for Toa'> <'step for ToA'> <'step for AoA'> <'number of AoA (default 180)'>
```
e.g., python Python_code/plot_combined.py 1.02e-08 2e-8 2e-10 1


## Python and relevant libraries version
Python >= 3.10.9
Numpy >= 1.23.5  
Scipy = 1.9.3  
Scikit-learn = 1.2.0  
OSQP >= 0.6.1


## Contact
Francesca Meneghello
meneghello@dei.unipd.it
github.com/francescamen