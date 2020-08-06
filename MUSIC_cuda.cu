/*
 * CUDA implementation of MUSIC DOA Estimation algorithm 
 * Hamza ERAY
 * Graduate School of Informatics - METU
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"
#include "helper_string.h"
#include <iostream>
#include <complex>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Core" 
#include <chrono>
#include <omp.h>
#include <cuComplex.h>	
#include <iomanip>


#define PI 3.14159265 

#define c 3e08 // speed of light

#define M 8 // number of antennas



std::vector<std::vector<std::complex<float>>> LoadSignal(char * sFileN, int SRow, int SCol);

void readAntennaPositions(char* fileName, Eigen::VectorXf &X_pos, Eigen::VectorXf &Y_pos, Eigen::VectorXf &Z_pos);

inline float deg2rad(float deg);

inline Eigen::VectorXcf steeringVector(float azimuth, float elevation, float signalFreq, Eigen::VectorXf X_pos, Eigen::VectorXf Y_pos, Eigen::VectorXf Z_pos);

Eigen::VectorXf fpeaks2D(Eigen::MatrixXf PP, float perc, Eigen::VectorXf &rowLocs, Eigen::VectorXf &colLocs, int numberOfThreads);

Eigen::VectorXf sort_peaks(Eigen::VectorXf peaks, Eigen::VectorXf & rowLocs, Eigen::VectorXf & colLocs, int D);



__global__ void MUSIC_kernel(cuFloatComplex *C_d_,
	const cuFloatComplex *__restrict__ A_table_d_,
	cuFloatComplex *invPrhs_table_d_, cuFloatComplex*invP_table_d_, float *P_table_d_,
	const int n_azim_, const int n_elev_, int TotalScan_)
{

	// Model using 1D grid structure 

	int idx, idy; // 2D-grid-compliant global thread indices
	int idGlb_1D = blockIdx.x * blockDim.x + threadIdx.x; // global 1D index for 2D -> 1D grid structure
        
	// azim & elev local indices by extracting idx & idy <-- 1D grid structure , idGlb_1D  
	// extract idx & idy from ( idGlb_1D = idx + idy * n_azim_ ) 
	idx = (idGlb_1D % n_azim_);   idy = (int)(idGlb_1D / n_azim_);


	/*Optim-step2 : Keeping the re-usable arrays in local memory
	C_h -> C_d -> C (on local-mem)
	Memory consid: for M=8; (4 - bytes) * 2 * (M * M) = 512 bytes*/

	cuFloatComplex C[M*M];

	for (int IndC = 0; IndC < M*M; IndC++)
	{
		C[IndC] = C_d_[IndC];
	}

	// Thread Indexing Bound-check  
	if (idGlb_1D < TotalScan_)
	{


		// printf("thread (%d,%d): i am alive \n", idx, idy); // activity check 
		

		// Common in-pass indexing for readibility &  better performance 

		int sclrInd = idx * 1 * 1 + idy * 1 * 1 * n_azim_; // single-pass scalar variable indexing, used for : invP_table_d , P_table_d
		int M_1_ind = idx * M * 1 + idy * M * 1 * n_azim_; // single-pass Mx1 array indexing, used for : A_table_d, invPrhs_table_d  

		
		/*
		 *  invPrhs = C (rw-by-slide) * a (slide-by-1)
		 */

		for (int cl = 0; cl < 1; cl++)
		{
			for (int rw = 0; rw < M; rw++)
			{
				cuFloatComplex slideSummer = make_cuComplex(0.0f, 0.0f);

				int sigSlid = cl * M + rw;

				for (int slide = 0; slide < M; slide++)

				{
					cuFloatComplex x = C[slide*M + rw];
					cuFloatComplex y = A_table_d_[(slide + cl * M) + M_1_ind];

					slideSummer.x = slideSummer.x + x.x * y.x - x.y * y.y;
					slideSummer.y = slideSummer.y + x.x * y.y + x.y * y.x;
				}
			
				// inserting the computed matrix multipl elements into allocated gmem array
				invPrhs_table_d_[(sigSlid)+M_1_ind] = slideSummer; // col-major (Mx1)
				
			}
		}


		/*
		 *  invP = a_adj ([rw=1]-by-slide) * invPrhs(slide-by-[cl=1])
		 */

		for (int cl = 0; cl < 1; cl++)
		{
			for (int rw = 0; rw < 1; rw++)
			{
				cuFloatComplex slideSummer = make_cuComplex(0.0f, 0.0f);

				int sigSlid = cl * M + rw;

				for (int slide = 0; slide < M; slide++)

				{
					cuFloatComplex x = cuConjf(A_table_d_[slide * 1 + rw + M_1_ind]);
					cuFloatComplex y = invPrhs_table_d_[(slide + cl * M) + M_1_ind];

					slideSummer.x = slideSummer.x + x.x * y.x - x.y * y.y;
					slideSummer.y = slideSummer.y + x.x * y.y + x.y * y.x;

				}

							
				// inserting the computed matrix multipl elements into allocated gmem array
				invP_table_d_[sclrInd] = slideSummer; // col-major (1x1)

			}
		}


		/*
		 *    Operation for P = 1/ invP.real()
		 */

		P_table_d_[sclrInd] = 1 / invP_table_d_[sclrInd].x;
		
	}

}



int main(int argc, char*argv[])
{

	int numberOfThreads = omp_get_max_threads() / 4;

	omp_set_num_threads(numberOfThreads); // set number of OpenMP threads (host side)

	std::cout << "Algorithm Properties:" << std::endl;
	std::cout << "Number of threads used by Eigen:" << Eigen::nbThreads() << std::endl;
	std::cout << "Number of threads used by OpenMP:" << omp_get_max_threads() << std::endl;

	// General parameters
	const int N = 1000; // number of samples
	int D = 2; // number of direct paths

        // SVD-related matrix size variables  
	const int m = M;
	const int n = M;
	const int lda = m;

	int BLKSIZE = 32; // block size (of CUDA threads)      

	
	float SignalFreqs = 15e6f; // RF carrier signal frequency
	
	float delta = 1.0f; // Azim/Elev angle step size

	
	float Start_Angle_A = 0.0f, Stop_Angle_A = 359.0f; // azimuth start & stop angle

	float Start_Angle_E = 01.0f, Stop_Angle_E = 90.0f; // elevation start & stop angle 

		
	const int n_azim = int((Stop_Angle_A - Start_Angle_A) / delta + 1); // number of azim angles
	const int n_elev = int((Stop_Angle_E - Start_Angle_E) / delta + 1); // number of elev angles 
	
	Eigen::VectorXf azimuth_series(n_azim);
	Eigen::VectorXf elevation_series(n_elev);
	azimuth_series.setLinSpaced(n_azim, Start_Angle_A, Stop_Angle_A); // linearly-spaced azim angle series  
	elevation_series.setLinSpaced(n_elev, Start_Angle_E, Stop_Angle_E); // linearly-spaced elev angle series

	std::cout << "n_azim: " <<  n_azim << std::endl << "n_elev: " << n_elev << std::endl;

	
	// Reading antenna positions
	Eigen::VectorXf X_rand(M), Y_rand(M), Z_rand(M);
	readAntennaPositions((char*)"TestData/ArrayPositions_Circ10m.txt", X_rand, Y_rand, Z_rand); // change here with your antenna config file name 
	std::cout << "antLoc read successful "  << std::endl;

	/*
	 * Reading synthetic/real signal from a text-file
	 */

	std::vector< std::vector<std::complex<float>>> X_prev;
	Eigen::MatrixXcf X(M, N);
	for (int i = 0; i < N; i++)
	{
		X_prev.push_back(std::vector<std::complex<float>>(M));
	}

	X_prev = LoadSignal((char*)"TestData/TestSignal_AWGN_15dB_150200_9090.txt", M, N); // change here with your input test signal file name

	

	std::cout << "signal read successful " << std::endl;

	for (int i = 0; i < M; i++) {  // EIGEN Conversion For Received Signal (X)

		for (int j = 0; j < N; j++)
		{
			X(i, j) = X_prev[i][j];
		}
	}


	// Timing text file creation 
	std::ofstream timerFile;
	std::string filename ="BLKsize"+ std::to_string(BLKSIZE) +  "_timerCUDA_N_" + std::to_string(N) + "Azimuth_" + std::to_string(int(Start_Angle_A)) + "_" +
	std::to_string(int(Stop_Angle_A)) + "Elevation_" + std::to_string(int(Start_Angle_E)) + "_" +
	std::to_string(int(Stop_Angle_E)) + "numOmpThreads_" + std::to_string(numberOfThreads) + ".txt";
 

	timerFile.open(filename, std::fstream::app);

	float tRavg = 0.0f;  float tJSVDavg = 0.0f; float tNSSavg = 0.0f; float tAh2davg = 0.0f; 
	float tArrprepavg = 0.0f;  float tAllavg = 0.0f; float tPd2havg = 0.0f;  float tFPKSavg = 0.0f;
	

	int numOfIteration = 1000; 

	Eigen::VectorXf azimuth_results;
	Eigen::VectorXf elevation_results;
	Eigen::VectorXf peak_values;

        /*
	 *  Uncomment here if you want to export MUSIC pseudo-spectrum values to a text file   
	 */ 
	/*std::ofstream music_cost;
	std::string filenameCost = "musicCost_" + std::to_string(N) + "Azimuth_" + std::to_string(Start_Angle_A) + "_" + std::to_string(Stop_Angle_A) + "Elevation_" + std::to_string(Start_Angle_E) + "_" + std::to_string(Stop_Angle_E) + ".txt";
	music_cost.open(filenameCost, std::fstream::app);*/
	
         /*
	  * Average run time is computed using 1000 run samples(named as iterations)  
 	  */

	for (int iteration = 0; iteration < numOfIteration; iteration++) {

	auto startR = std::chrono::high_resolution_clock::now();
	
	Eigen::MatrixXcf X_adj(N, M);
	Eigen::MatrixXcf R_hat(M, M);
	X_adj = X.adjoint();
	R_hat = (X * X_adj) / N;


	auto diffR = std::chrono::high_resolution_clock::now() - startR;
	auto tR = std::chrono::duration_cast<std::chrono::microseconds>(diffR);

	tRavg += tR.count();

		
	cuFloatComplex * R_hat_h = NULL;

	R_hat_h = new cuFloatComplex[lda*n];
		
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < lda; i++)
		{
			R_hat_h[j*lda + i].x = R_hat(i, j).real();
			R_hat_h[j*lda + i].y = R_hat(i, j).imag();
		}

	}

	auto startJSVD = std::chrono::high_resolution_clock::now();	
	
	/*
	 * SVD computation using cuSOLVER 
	 */
	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream = NULL;
	gesvdjInfo_t gesvdj_params = NULL;

	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;


	cuFloatComplex * U_h = NULL;
	U_h = new cuFloatComplex[lda*m]; /* m-by-m unitary matrix, left singular vectors  */

	cuFloatComplex * V_h = NULL;
	V_h = new cuFloatComplex[lda*n]; /* n-by-n unitary matrix, right singular vectors */

	float *S_h = NULL;
	S_h = new float[n];  /* numerical singular value */



	cuFloatComplex *R_hat_d = NULL;  /* device copy of R_hat */
	float *S_d = NULL;  /* singular values */
	cuFloatComplex *U_d = NULL;  /* left singular vectors */
	cuFloatComplex *V_d = NULL;  /* right singular vectors */
	int *d_info = NULL;  /* error info */
	int lwork = 0;       /* size of workspace */
	cuFloatComplex *d_work = NULL; /* device workspace for gesvdj */
	int info = 0;        /* host copy of error info */



/* configuration of gesvdj  */
	const double tol = 1.e-7;
	const int max_sweeps = 15;
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
	const int econ = 0; /* econ = 1 for economy size */

/* numerical results of gesvdj  */
	double residual = 0;
	int executed_sweeps = 0;

	/* step 1: create cusolver handle, bind a stream */
	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cudaStat1);

	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: configuration of gesvdj */
	status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of tolerance is machine zero */
	status = cusolverDnXgesvdjSetTolerance(
		gesvdj_params,
		tol);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of max. sweeps is 100 */
	status = cusolverDnXgesvdjSetMaxSweeps(
		gesvdj_params,
		max_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 3: copy A and B to device */
	cudaStat1 = cudaMalloc((void**)&R_hat_d, sizeof(std::complex<float>)*lda*n);
	cudaStat2 = cudaMalloc((void**)&S_d, sizeof(float)*n);
	cudaStat3 = cudaMalloc((void**)&U_d, sizeof(std::complex<float>)*lda*m);
	cudaStat4 = cudaMalloc((void**)&V_d, sizeof(std::complex<float>)*lda*n);
	cudaStat5 = cudaMalloc((void**)&d_info, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	assert(cudaSuccess == cudaStat5);

	cudaStat1 = cudaMemcpy(R_hat_d, R_hat_h, sizeof(std::complex<float>)*lda*n, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);

	/* step 4: query workspace of SVD */
	status = cusolverDnCgesvdj_bufferSize(
		cusolverH,
		jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
			  /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
		econ, /* econ = 1 for economy size */
		m,    /* nubmer of rows of A, 0 <= m */
		n,    /* number of columns of A, 0 <= n  */
		R_hat_d,  /* m-by-n */
		lda,  /* leading dimension of A */
		S_d,  /* min(m,n) */
			  /* the singular values in descending order */
		U_d,  /* m-by-m if econ = 0 */
			  /* m-by-min(m,n) if econ = 1 */
		lda,  /* leading dimension of U, ldu >= max(1,m) */
		V_d,  /* n-by-n if econ = 0  */
			  /* n-by-min(m,n) if econ = 1  */
		lda,  /* leading dimension of V, ldv >= max(1,n) */
		&lwork,
		gesvdj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuFloatComplex)*lwork);
	assert(cudaSuccess == cudaStat1);

	/* step 5: compute SVD */
	status = cusolverDnCgesvdj(
		cusolverH,
		jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
			   /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
		econ,  /* econ = 1 for economy size */
		m,     /* nubmer of rows of A, 0 <= m */
		n,     /* number of columns of A, 0 <= n  */
		R_hat_d,   /* m-by-n */
		lda,   /* leading dimension of A */
		S_d,   /* min(m,n)  */
			   /* the singular values in descending order */
		U_d,   /* m-by-m if econ = 0 */
			   /* m-by-min(m,n) if econ = 1 */
		lda,   /* leading dimension of U, ldu >= max(1,m) */
		V_d,   /* n-by-n if econ = 0  */
			   /* n-by-min(m,n) if econ = 1  */
		lda,   /* leading dimension of V, ldv >= max(1,n) */
		d_work,
		lwork,
		d_info,
		gesvdj_params);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMemcpy(U_h, U_d, sizeof(cuFloatComplex)*lda*m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V_h, V_d, sizeof(cuFloatComplex)*lda*n, cudaMemcpyDeviceToHost); /*not-have-to-be-copied to the host-side, not-used in MUSIC computations*/
	cudaStat3 = cudaMemcpy(S_h, S_d, sizeof(float)*n, cudaMemcpyDeviceToHost);
	cudaStat4 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	cudaStat5 = cudaDeviceSynchronize();
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	assert(cudaSuccess == cudaStat5);

	auto diffJSVD = std::chrono::high_resolution_clock::now() - startJSVD;
	auto tJSVD = std::chrono::duration_cast<std::chrono::microseconds>(diffJSVD);

	tJSVDavg += tJSVD.count();
		
	if (0 == info) {
		printf("gesvdj converges \n");
	}
	else if (0 > info) {
		printf("%d-th parameter is wrong \n", -info);
		exit(1);
	}
	else {
		printf("WARNING: info = %d : gesvdj does not converge \n", info); // If svd not converge, program warns the user 
	}

	status = cusolverDnXgesvdjGetSweeps(
		cusolverH,
		gesvdj_params,
		&executed_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	status = cusolverDnXgesvdjGetResidual(
		cusolverH,
		gesvdj_params,
		&residual);
	assert(CUSOLVER_STATUS_SUCCESS == status);

		

	/*
	 *  1D ARRAY to EIGEN MATRIX TRANSFORMS
	 */


	Eigen::MatrixXcf U(lda, m);
	Eigen::MatrixXcf V(lda, n);
	Eigen::VectorXf S(n);


	for (int col = 0; col < m; col++)
	{
		for (int row = 0; row < lda; row++)
		{
			U(row, col) = std::complex<float>(U_h[row + col * lda].x, U_h[row + col * lda].y);
		}
	}

	for (int col = 0; col < 1; col++)
	{
		for (int row = 0; row < n; row++)
		{
			S(row, col) = S_h[row + col * n];
		}
	}


	/*
	 * Signal Subspace / Noise Subspace Selection
	 */

	auto startNSS = std::chrono::high_resolution_clock::now();

	Eigen::MatrixXcf NU = Eigen::MatrixXcf::Zero(M, M - D);
	Eigen::MatrixXcf NU_adj = Eigen::MatrixXcf::Zero(M - D , M);
	Eigen::MatrixXcf C(M, M);

	// Block of size (p,q), starting at (i,j)	matrix.block(i, j, p, q);

	NU.block(0, 0,
		M, M - D)
		<<
		U.block(0, D,
			    M, M - D);


	NU_adj = NU.adjoint();
	C = NU * NU_adj;

	auto diffNSS = std::chrono::high_resolution_clock::now() - startNSS;
	auto tNSS = std::chrono::duration_cast<std::chrono::microseconds>(diffNSS);

	tNSSavg += tNSS.count();


		
	cuFloatComplex * C_h = NULL;

	cudaMallocHost((void**)&C_h, sizeof(cuFloatComplex)*M*M);

		
	for (int col = 0; col < M; col++)
	{
		for (int row = 0; row < M; row++)
		{
			C_h[row + col * M].x = C(row, col).real();
			C_h[row + col * M].y = C(row, col).imag();
		}
	}

	cuFloatComplex * C_d = NULL;
	cudaError_t cudaStatCalloc = cudaSuccess;
	cudaStatCalloc = cudaMalloc((void**)&C_d, sizeof(cuFloatComplex)*M*M);
	assert(cudaSuccess == cudaStatCalloc);

	cudaError_t cudaStatCh2d = cudaSuccess;
	cudaStatCh2d = cudaMemcpy(C_d, C_h, sizeof(cuFloatComplex)*M*M, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStatCh2d);



	auto startArrPrep = std::chrono::high_resolution_clock::now();
		
	/*
	 * Generation of the Array Manifold via Steering vectors
	 * && 
	 * Constructing the reusable A_table for all CUDA threads
	 */


	cuFloatComplex *A_table_h = NULL;

	cudaMallocHost((void**)&A_table_h, sizeof(cuFloatComplex)*M * n_azim * n_elev);
		

	Eigen::MatrixXcf* arrayManifold = new Eigen::MatrixXcf[n_azim];
	for (int i = 0; i < n_azim; i++)
	{
		arrayManifold[i] = Eigen::MatrixXcf(M, n_elev);
	}



	for (int elevInd = 0; elevInd < n_elev; elevInd++)
	{
		for (int azimInd = 0; azimInd < n_azim; azimInd++)
		{
			float az = azimuth_series(azimInd);
			float el = elevation_series(elevInd);
			arrayManifold[azimInd].col(elevInd) = steeringVector(az, el, SignalFreqs, X_rand, Y_rand, Z_rand);

			for (int steerInd = 0; steerInd < M; steerInd++)
			{
				float temp_Re, temp_Im;

				temp_Re = arrayManifold[azimInd](steerInd, elevInd).real();

				temp_Im = arrayManifold[azimInd](steerInd, elevInd).imag();

				A_table_h[elevInd*M*n_azim + azimInd * M + steerInd] = make_cuComplex(temp_Re, temp_Im); // col-major indexing
			}
		}
	}


	auto diffArrPrep = std::chrono::high_resolution_clock::now() - startArrPrep;
	auto tArrprep = std::chrono::duration_cast<std::chrono::microseconds>(diffArrPrep);

	tArrprepavg += tArrprep.count();


		
	// definition for the time performance measure-A_table_alloc & mem-cpy
	cudaEvent_t startA, stopA;
	float estTimeA;
	cudaEventCreate(&startA);
	cudaEventCreate(&stopA);

	// start the timer for A_table  
	cudaEventRecord(startA, 0);
	

	cuFloatComplex * A_table_d = NULL;
	cudaError_t cudaStatAtable = cudaSuccess;
	cudaStatAtable = cudaMalloc((void**)&A_table_d, sizeof(cuFloatComplex) * M * n_azim * n_elev);
	assert(cudaSuccess == cudaStatAtable);
		

	cudaError_t cudaStatAtableH2D = cudaSuccess;

	cudaStatAtableH2D = cudaMemcpy(A_table_d, A_table_h, sizeof(cuFloatComplex) * M * n_azim * n_elev, cudaMemcpyHostToDevice);

	assert(cudaSuccess == cudaStatAtableH2D);

	// stop the timer for A_table
	cudaEventRecord(stopA, 0);
	cudaEventSynchronize(stopA);
	cudaEventElapsedTime(&estTimeA, startA, stopA);

	tAh2davg += estTimeA; 

	/*
	 * Allocation for kernel-related arrays (to realize in-kernel operations using global memory arrays)
	 */

	float *P_table_h = NULL;

	cudaMallocHost((void**)&P_table_h, sizeof(float)* 1 * n_azim * n_elev);
		
		
		
	for (int Pth= 0; Pth < n_azim * n_elev; Pth++)
	{
		P_table_h[Pth] = 0.0f;
	}


	cuFloatComplex * invPrhs_table_d = NULL;
	cudaError_t cudaStatinvPrhstable = cudaSuccess;
	cudaStatinvPrhstable = cudaMalloc((void**)&invPrhs_table_d, sizeof(cuFloatComplex) * M * 1 * n_azim * n_elev);
	assert(cudaSuccess == cudaStatinvPrhstable);


	cuFloatComplex * invP_table_d = NULL;
	cudaError_t cudaStatinvPtable = cudaSuccess;
	cudaStatinvPtable = cudaMalloc((void**)&invP_table_d, sizeof(cuFloatComplex) * 1 * 1 * n_azim * n_elev);
	assert(cudaSuccess == cudaStatinvPtable);


	float * P_table_d = NULL;
	cudaError_t cudaStatPtable = cudaSuccess;
	cudaStatPtable = cudaMalloc((void**)&P_table_d, sizeof(float) * 1 * 1 * n_azim * n_elev);
	assert(cudaSuccess == cudaStatPtable);


	cudaError_t cudaStatPtableh2d = cudaSuccess;

	cudaStatPtableh2d = cudaMemcpy(P_table_d, P_table_h, sizeof(float) * 1 * n_azim * n_elev, cudaMemcpyHostToDevice);

	assert(cudaSuccess == cudaStatPtableh2d);

				
	

	/*
	 *  MUSIC kernel-configuration
	 */

	int TotalScan = n_azim * n_elev; // total number of active threads to be spawn
 
	int blNum = (TotalScan) / BLKSIZE + (TotalScan % BLKSIZE == 0 ? 0 : 1);


	cudaEvent_t startKernel, stopKernel;
	float estTimeMUSICkernel;
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);

	// start the timer for MUSIC kernel  
	cudaEventRecord(startKernel, 0);
		
        /*
	 *  MUSIC kernel-launch with preferred configuration 
	 */    
	MUSIC_kernel << < blNum, BLKSIZE >> > (C_d, A_table_d,
		invPrhs_table_d, invP_table_d, P_table_d,
		n_azim, n_elev, TotalScan);


	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("MUSIC Kernel-launch CUDA Error: %s\n", cudaGetErrorString(err));
	}


	cudaEventRecord(stopKernel, 0);
	cudaEventSynchronize(stopKernel);
	cudaEventElapsedTime(&estTimeMUSICkernel, startKernel, stopKernel);

	tAllavg += estTimeMUSICkernel; 


	// definition for the time performance of resultant P_table D2H mem-cpy
	cudaEvent_t startPd2h, stopPd2h;
	float estTimepd2h;
	cudaEventCreate(&startPd2h);
	cudaEventCreate(&stopPd2h);

	// start the timer for A_table  
	cudaEventRecord(startPd2h, 0);
		
	cudaError_t cudaStatPd2h = cudaSuccess;

	cudaStatPd2h = cudaMemcpy(P_table_h, P_table_d, sizeof(float) * 1 * 1 * n_azim * n_elev, cudaMemcpyDeviceToHost);

	assert(cudaSuccess == cudaStatPd2h);
	
	cudaError_t err2 = cudaGetLastError();
	if (err2 != cudaSuccess)
	{
		printf("P_table D2H memcpy CUDA Error: %s\n", cudaGetErrorString(err));
	}

	// stop the timer for P_table D2H mem-cpy
	cudaEventRecord(stopPd2h, 0);
	cudaEventSynchronize(stopPd2h);
	cudaEventElapsedTime(&estTimepd2h, startPd2h, stopPd2h);


	tPd2havg += estTimepd2h;


		

	Eigen::MatrixXf P(n_azim, n_elev);


	for (int i = 0; i < n_azim; i++)
	{
		for (int j = 0; j < n_elev; j++)
		{
			P(i, j) = P_table_h[i + j * n_azim];
		}
	}

	/*
	 *  Uncomment if you want to realize numerical precision test (via exporting into a text file)
	 */

	/*
	for (int j = 0; j < n_elev; j++)
	{
		for (int i = 0; i < n_azim; i++)
		{
			music_cost << P(i, j) << std::setprecision(8) << std::endl;
		}
	}

	music_cost.close();
	*/	
		
	auto startFPKS = std::chrono::high_resolution_clock::now();

	Eigen::VectorXf azimuth_locs;
	Eigen::VectorXf elevation_locs;


	peak_values = fpeaks2D(P, 0, azimuth_locs, elevation_locs, numberOfThreads);

	peak_values = sort_peaks(peak_values, azimuth_locs, elevation_locs, D);


	auto diffFPKS = std::chrono::high_resolution_clock::now() - startFPKS;
	auto tFPKS = std::chrono::duration_cast<std::chrono::microseconds>(diffFPKS);

	tFPKSavg += tFPKS.count();
		
	elevation_results = (elevation_locs.array()) * delta + Start_Angle_E;
	azimuth_results = (azimuth_locs.array()) * delta + Start_Angle_A;

	/*  Free Host resources  */

	if (R_hat_h)
	{
		delete[] R_hat_h; // When done, free memory pointed to by R_hat_h.
		R_hat_h = NULL;     // Clear R_hat_h to prevent using invalid memory reference.
	}

	if (S_h)
	{
		delete[] S_h;
		S_h = NULL;
	}

	if (U_h)
	{
		delete[] U_h;
		U_h = NULL;
	}

	if (V_h)
	{
		delete[] V_h;
		V_h = NULL;
	}


	/* Free the pinned host allocations */

		if (C_h)
	{
		cudaFreeHost(C_h);
		C_h = NULL;
	}

	if (A_table_h)
	{
		cudaFreeHost(A_table_h);
		A_table_h = NULL;
	}

	if (P_table_h)
	{
		cudaFreeHost(P_table_h);
		P_table_h = NULL;
	}


		
	/*  Free Device resources  */
	if (R_hat_d) cudaFree(R_hat_d);
	if (S_d) cudaFree(S_d);
	if (U_d) cudaFree(U_d);
	if (V_d) cudaFree(V_d);
	if (d_info) cudaFree(d_info);
	if (d_work) cudaFree(d_work);


	if (C_d) cudaFree(C_d);

	if (A_table_d) cudaFree(A_table_d);

	if (invPrhs_table_d) cudaFree(invPrhs_table_d);
	if (invP_table_d) cudaFree(invP_table_d);
	if (P_table_d) cudaFree(P_table_d);


	if (cusolverH) cusolverDnDestroy(cusolverH);
	if (stream) cudaStreamDestroy(stream);
	if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);

	cudaDeviceReset();
	}

	// averaging total durations for each step in the code flow   
	tRavg = tRavg / float(numOfIteration);  tJSVDavg = tJSVDavg / float(numOfIteration);  tNSSavg = tNSSavg / float(numOfIteration); tAh2davg = tAh2davg / float(numOfIteration);
	tArrprepavg = tArrprepavg/float(numOfIteration);  tAllavg = tAllavg / float(numOfIteration); tPd2havg = tPd2havg/ float(numOfIteration);  tFPKSavg = tFPKSavg / float(numOfIteration);


        // Writing the average part-durations into the out stream initiated for the timing text file   
	timerFile << "tR: " << tRavg << " usec " << std::endl
		<< "tJSVD: " << tJSVDavg << " usec " << std::endl
		<< "tNSS: " << tNSSavg << " usec " << std::endl
		<< "tArrprep: " << tArrprepavg << " usec " << std::endl
		<< "tAh2d: " << tAh2davg << " msec " << std::endl
		<< "tAll: " << tAllavg << " msec " << std::endl
		<< "tPd2h: " << tPd2havg << " msec " << std::endl	
		<< "tFPKS: " << tFPKSavg << " usec " << std::endl;


	timerFile.close();


        /*
	 * Program finalization & algorithm result messages  
	 */
	std::cout << "Durations are recorded! " << std::endl;
	std::cout << "Azimuth:" << azimuth_results.transpose() << std::endl;
	std::cout << "Elevation:" << elevation_results.transpose() << std::endl;
	std::cout << "Peaks:" << peak_values.transpose() << std::endl;

}


std::vector< std::vector<std::complex<float>>>
LoadSignal(char * sFileN, int SRow, int SCol)
{
	FILE *file;
	errno_t err;
	std::vector< std::vector<std::complex<float>>> n;

	for (int rwInd = 0; rwInd < SRow; ++rwInd)
	{
		n.push_back(std::vector<std::complex<float>>(SCol));
	}
        
	/*
	 * temporary(empty) array & Real/Imaginary arrays 
	 * their sizes are constant for now, to be changed to variables later  
	 */
	int Empty[8000];
	float Real[8][1000];
	float Imag[8][1000];


	
	err = fopen_s(&file, sFileN, "r");

	if (err == 0)
	{
		int i = 0;
		while (!feof(file))
		{
			for (int rs = 0; rs < SRow; rs++)
			{

				for (int cs = 0; cs < SCol; cs++)
				{

					fscanf_s(file, "%f\t%f", &Real[rs][cs], &Imag[rs][cs]);

					if (cs < 999) // end of the line (set to N-1 value if N is not 1000)
						fscanf_s(file, "\t\t", &Empty[i]);
					i++;
				}
			}
		}
	}

	else
	{
		printf("The requested file could not be opened\n");
	}

	for (int Sum = 0; Sum < SRow; Sum++)
		for (int sumR = 0; sumR < SCol; sumR++) {
			n[Sum][sumR] = std::complex<float>(Real[Sum][sumR], Imag[Sum][sumR]);
		}

	return n;
	fclose(file);
}


void readAntennaPositions(char* fileName, Eigen::VectorXf &X_pos, Eigen::VectorXf &Y_pos, Eigen::VectorXf &Z_pos) {
	FILE *file;
	fopen_s(&file, fileName, "r");
	int i = 0;
	while (!feof(file)) {
		int antennaNumber;
		fscanf_s(file, "a%d:%f_%f_%f;\n", &antennaNumber, &X_pos[i], &Y_pos[i], &Z_pos[i]);
		i++;
	}
	fclose(file);
}



inline float deg2rad(float deg) {
	float rad = PI * deg / 180.0f;
	return rad;
}


inline Eigen::VectorXcf steeringVector(float azimuth, float elevation, float signalFreq, Eigen::VectorXf X_pos, Eigen::VectorXf Y_pos, Eigen::VectorXf Z_pos) {
	int rowX = X_pos.size();
	Eigen::VectorXcf a(rowX);
	for (int i = 0; i < M; i++) {
		std::complex<float> insideExp(0, 0);
		float A = X_pos[i] * cos(deg2rad(azimuth))*sin(deg2rad(elevation));
		float B = Y_pos[i] * sin(deg2rad(elevation))*sin(deg2rad(azimuth));
		float C = Z_pos[i] * cos(deg2rad(elevation));
		insideExp = std::complex<float>(0, 2 * PI*(1 / c)*signalFreq*(A + B + C));
		a[i] = exp(insideExp);
	}
	return a;
}


Eigen::VectorXf fpeaks2D(Eigen::MatrixXf PP, float perc, Eigen::VectorXf &rowLocs, Eigen::VectorXf &colLocs, int numberOfThreads) {

	Eigen::MatrixXf P = Eigen::MatrixXf::Zero(PP.rows() + 2, PP.cols() + 2);
	P.block(1, 1, PP.rows(), PP.cols()) = PP;

	int colSize = P.cols();
	int rowSize = P.rows();
	rowLocs.setZero();
	colLocs.setZero();

	// Thresholding...
	float threshold = perc * P.maxCoeff();

	// Diff calculations...
	Eigen::MatrixXf P_diff_C(rowSize, colSize);
	Eigen::MatrixXf P_diff_R(rowSize, colSize);
	P_diff_C.setZero();
	P_diff_R.setZero();
	Eigen::VectorXf peaks(rowLocs.rows());
	peaks.setZero();
	int i;
	int j;

#pragma omp parallel private(i,j)  num_threads(numberOfThreads)
	{
#pragma omp for
		for (i = 0; i < max(colSize, rowSize) - 1; i++) {
			if (i < colSize - 1)
				P_diff_C.col(i) = P.col(i) - P.col(i + 1);
			if (i < rowSize - 1)
				P_diff_R.row(i) = P.row(i) - P.row(i + 1);
		}

		// Peak sellection...
		bool isOneRow = (rowSize == 1);
		bool isOneCol = (colSize == 1);
		int index = 0;

		//#pragma omp for collapse(2) 
#pragma omp for 
		for (i = (1 - isOneRow); i < rowSize; i++) {
			for (j = (1 - isOneCol); j < colSize; j++) {
				bool isPeakVert = (P_diff_R(i - (1 - isOneRow), j) < 0 && P_diff_R(i, j) > 0);
				bool isPeakHorz = (P_diff_C(i, j - (1 - isOneCol)) < 0 && P_diff_C(i, j) > 0);

				if (((P(i, j) > threshold)) && ((isOneRow && isPeakHorz) || (isOneCol&& isPeakVert) || (isPeakVert && isPeakHorz)))
				{
#pragma omp critical
					{
						peaks.conservativeResize(peaks.rows() + 1);
						colLocs.conservativeResize(colLocs.rows() + 1);
						rowLocs.conservativeResize(rowLocs.rows() + 1);
						peaks(peaks.rows() - 1) = P(i, j);
						colLocs(colLocs.rows() - 1) = j - 1;
						rowLocs(rowLocs.rows() - 1) = i - 1;
					}
				}
			}
		}
	}

	return peaks;
}



/*
 * Sorts the DF log peak values from high value to low value.
 * Returns maximum of D count of peaks
 */
Eigen::VectorXf sort_peaks(Eigen::VectorXf peaks, Eigen::VectorXf & rowLocs, Eigen::VectorXf & colLocs, int D)
{
	Eigen::VectorXf colLocs_temp = colLocs;
	Eigen::VectorXf rowLocs_temp = rowLocs;
	Eigen::VectorXf colLocs_all = colLocs;
	Eigen::VectorXf rowLocs_all = rowLocs;
	Eigen::VectorXf sorted_peaks = peaks;

	std::sort(sorted_peaks.data(), sorted_peaks.data() + sorted_peaks.rows(), std::greater<float>());

	int D_min = min(D, static_cast<int>(peaks.rows()));
	for (int i = 0; i < D_min; i++)
	{
		for (int j = 0; j < peaks.rows(); j++)
		{
			if (sorted_peaks(i) == peaks(j))
			{
				colLocs_temp(i) = colLocs_all(j);
				rowLocs_temp(i) = rowLocs_all(j);
				break;
			}
		}
	}
	colLocs = colLocs_temp.head(D_min);
	rowLocs = rowLocs_temp.head(D_min);
	peaks = sorted_peaks.head(D_min);
	return peaks;
}
