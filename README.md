# NssDOACuda
Noise Subspace(NSS)-based DOA Estimation Algorithms on GPU using CUDA

Most popular algorithms in this family are named as follows.  
* PHD (Pisarenko Harmonic Decomposition)
*	MUSIC (MUltiple SIgnal Classification)
*	EV (Eigen Vector)
*	MN (Minimum Norm) 

Since they are similar to each other in terms of numerical & duration performance, only the MUSIC algorithm is provided within this repository.

The companion MUSIC code is provided for the following preprint:

* name & arXiv link are to be added after arXiv uploading *

## Test Scenario 

A specific DOA test scenario has been designed for numerical validation and experimental evaluation of the performance of the algorithms. The scenario assumes that two RF sources are uncorrelated, incoming signals are narrowband and carried at 15 MHz with 15 dB SNR level. Eight omnidirectional antennas are positioned in a uniformly circular way with a radius of 10 m. It is depicted in the following figure. 

<p align="center">
  <img src="https://github.com/erayhamza/NssDOACuda/blob/master/images/TestScenario.JPG" width="500" height="auto">
</p>

## Requirements

This implementation uses Eigen template library for some host-side linear algebra operations. However, depending on the version, the content of Eigen library sometimes conflicts with CUDA toolkit and this may require additional effort for discarding some unused folders from the root Eigen library or for doing some additional Visual Studio project settings.    


## Commands

_Note: These implementations require NVIDIA CUDA Toolkit (tested with version 10.1) to be installed on the machine._

CUDA code compilation after setting some parameters & editing the related information about the input data :

```
nvcc .\MUSIC_cuda.cu
```

CUDA code running after compilation:

.\*.out file is run after making input data available to the code path 



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
