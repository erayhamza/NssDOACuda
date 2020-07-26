# NssDOACuda
Noise Subspace(NSS)-based DOA Estimation Algorithms on GPU using CUDA

This repository contains the following codes belonging to four NSS-DOA algorithms: 
* PHD (Pisarenko Harmonic Decomposition)
*	MUSIC (MUltiple SIgnal Classification)
*	EV (Eigen Vector)
*	MN (Minimum Norm) 

They are companion codes for the following preprint:

* name & arXiv link are to be added after arXiv uploading *

## Test Scenario 

A specific DOA test scenario has been designed for numerical validation and experimental evaluation of the performance of the algorithms. The scenario assumes that two RF sources are uncorrelated, incoming signals are narrowband and carried at 15 MHz with 15 dB SNR level. Eight omnidirectional antennas are positioned in a uniformly circular way with a radius of 10 m. It is depicted in the following figure. 

<p align="center">
  <img src="https://github.com/erayhamza/NssDOACuda/blob/master/images/TestScenario.JPG" width="500" height="auto">
</p>

## Commands

_Note: These implementations require NVIDIA CUDA Toolkit (tested with version 10.1) to be installed on the machine._

To compile:

```
All nvcc-based compile commands are to be added 
```

To run:

```
All run commands from out files are to be added  
```


## Citation

If you use this code, please cite the paper using the reference below:

> Citation info is to be added 

BibTeX entry:

```
BibTeX info is to be added 
```



## Credits

Inria TUX Family - Eigen template library (used in most of the linear algebra operations) :

*https://gitlab.com/libeigen/eigen*
