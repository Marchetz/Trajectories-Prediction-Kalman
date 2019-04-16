# Trajectories-Prediction-Kalman
Study and analysis of trajectories Prediction with Kalman Filter.
The file 'report.pdf' describes the work.

## Installation

The scripts are written in Python 3.6.

This project requires the following Python packages installed:
* numpy
* matplotlib

## Example execution

This command starts the trajectories prediction analysis using kalman filter with uniformly accelerated motion and save the qualitative results:
```
$ python main.py -s -a 
```
The details of analysis and qualitative results are saved in a folder.

For analysis described in chapter 5.4 in report.pdf, use the command:
```
$ python analysis_homography
```



## Command line arguments
```
    -h, --help                     show this help message and exit.
    -s, --save                     save the qualitative results.
    -p0 P0                         P0 diagonal value, the initial Process Covariance Matrix. (default: 0.03)
    -q Q                           Q diagonal value, the Process Noise Covariance Matrix. (default: 0.03)
    -r0 R0                         R0 diagonal value, the initial Measurements Noise Covariance Matrix. (default: 0.03)
    '-n', --past_len               past length (default: 20)
    -a, --acceleration             use acceleration (default: False)

```

**Note**:
This project has been developed for the course "Image and Video Analysis" ( Università degli studi di Firenze ).
It has been resumed the work made by Simone Magistri and Ivan Prosperi. I thank them for excellent work.


## Authors
* **Francesco Marchetti**
