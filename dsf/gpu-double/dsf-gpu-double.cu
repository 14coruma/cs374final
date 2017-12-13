/* dsf-gpu-double.cu is a prime factorization program using Direct Search Factorization
 * and CUDA to parallelize and double precision.
 * This is good for factoring up to about 15 digits.
 *
 * Andrew Corum, Dec 2017
 * Usage: dsf [num] [num] [table format?]
 */

#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <gmp.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__device__
void nextprime(double *prime) {
	if(*prime < 2) {
		*prime = 2.0;
	} else if (fmod(*prime, 2.0) == 0.0) {
		*prime += 1.0;
	} else {
		*prime += 2.0;
	}
}

/**
 * CUDA Kernel Device code
 */
__global__
void vectorHypot(
	const double root, const double N,
	double *f1, double *f2, double numThreads
)
{
	__shared__ int done;
	done = 0;
	double id = blockDim.x * blockIdx.x + threadIdx.x;
	double prime, max;

	// Figure out starting prime
	prime = ceil(__ddiv_ru(root, numThreads));
	max = prime * (id + 1);
	prime *= id;
	nextprime(&prime);

	// Loop through potential factors
	while (prime <= max && done == 0) {
		// If prime divides N, add prime and q to factors, then break
		if (fmod(N, prime) == 0.0) {
			f1[0] = prime;
			f2[0] = __ddiv_rd(N, prime);
			done = 1;
			break;
		} else {
			// Otherwise figure out next prime and continue
			nextprime(&prime);
		}
	}
}

void checkErr(cudaError_t err, const char* msg) 
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s (error code %d: '%s')!\n", msg, err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Host main routine
 */
int main(int argc, char** argv)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate the host input vectors A & B
	double root, N, N1, N2;
	size_t size = sizeof(double);
	N1 = atoi(argv[1]);
	N2 = atoi(argv[2]);
	N = N1 * N2;
	root = sqrt(N);

	double * h_f1 = NULL;
	h_f1 = (double *) malloc(size);
	double * h_f2 = NULL;
	h_f2 = (double *) malloc(size);

    // Verify that allocations succeeded
    if (h_f1 == NULL || h_f1 == NULL || root == NULL || N == NULL)
    {
        fprintf(stderr, "Failed to allocate host values!\n");
        exit(EXIT_FAILURE);
    }

    // 1a. Allocate the device input vectors A & B
    double *d_f1 = NULL;
    err = cudaMalloc((void **)&d_f1, size);
    checkErr(err, "Failed to allocate device value d_f1");
    double *d_f2 = NULL;
    err = cudaMalloc((void **)&d_f2, size);
    checkErr(err, "Failed to allocate device value d_f2");

    /*// 1.b. Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    checkErr(err, "Failed to allocate device vector C");*/

	double wtime = -omp_get_wtime();
    // 2. Copy the host input vectors A and B in host memory 
    //     to the device input vectors in device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_f1, h_f1, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device value d_f1 from host to device");


    err = cudaMemcpy(d_f2, h_f2, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device value d_f2 from host to device");

    // 3. Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 7;
    double numThreads = threadsPerBlock * blocksPerGrid;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorHypot<<<blocksPerGrid, threadsPerBlock>>>(root, N, d_f1, d_f2, numThreads);
    err = cudaGetLastError();
    checkErr(err, "Failed to launch vectorHypot kernel");

    // 4. Copy the device result vector in device memory
    //     to the host result vector in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_f1, d_f1, size, cudaMemcpyDeviceToHost);
    checkErr(err, "Failed to copy value h_f1 from device to host");
    err = cudaMemcpy(h_f2, d_f2, size, cudaMemcpyDeviceToHost);
    checkErr(err, "Failed to copy value h_f2 from device to host");
	wtime += omp_get_wtime();

/*    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(sqrt(h_A[i] * h_A[i] + h_B[i] * h_B[i]) - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }*/
	//printf("(%lu x %lu)\t(%lu x %lu)\n", N1, N2, *h_f1, *h_f2);
    //printf("CUDA test PASSED\n");
	printf("%.0f\t%f\t(%.0f x %.0f)\n", ceil(log10(N)), wtime, *h_f1, *h_f2);

    // Free device global memory
    err = cudaFree(d_f1);
    checkErr(err, "Failed to free device vector A");

    err = cudaFree(d_f2);
    checkErr(err, "Failed to free device vector B");

    // Free host memory
    free(h_f1);
    free(h_f2);

    // Reset the device and exit
    err = cudaDeviceReset();
    checkErr(err, "Unable to reset device");

    return 0;
}

