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
#include "cubi.h"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__device__
void nextprime(cubi prime) {
	cubi one, two, prime_copy, R, Q;
	one        = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	two        = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	prime_copy = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	R          = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	Q          = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	cubi_init_d(prime_copy);
	cubi_init_d(one);
	cubi_init_d(two);
	cubi_init_d(R);
	cubi_init_d(Q);

	cubi_copy_d(prime, prime_copy);
	one[0] = 1;
	two[0] = 2;

	if(cubi_cmp_d(prime, two) < 0) {
		cubi_copy_d(two, prime);
	} else {
		cubi_div_d(prime, two, Q, R);
		if (cubi_cmp_d(R, one) != 0) {
			cubi_add_d(prime_copy, one, prime);
		} else {
			cubi_add_d(prime_copy, two, prime);
		}
	}

	free(one);
	free(two);
	free(prime_copy);
	free(R);
	free(Q);
}

/**
 * CUDA Kernel Device code
 */
__global__
void vectorHypot(
	cubi d_root, cubi d_N,
	cubi d_f1, cubi d_f2, cubi d_numThreads
)
{
	__shared__ int done;
	done = 0;

	// Creat cubis for all parameters
	cubi f1, f2, root, N, numThreads;
	root       = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	N          = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	f1         = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	f2         = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	numThreads = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	cubi_init_d(f1);
	cubi_init_d(f2);
	cubi_init_d(root);
	cubi_init_d(N);
	cubi_init_d(numThreads);

	for (int i = 0; i < SIZE; i++) {
		f1[i] = d_f1[i];
		f2[i] = d_f2[i];
		root[i] = d_root[i];
		N[i] = d_N[i];
	}
	numThreads[0] = d_numThreads[0];

	cubi id, prime, prime_copy, max, zero, R, Q;
	id         = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	prime      = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	prime_copy = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	max        = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	zero       = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	R          = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	Q          = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	cubi_init_d(id);
	cubi_init_d(prime);
	cubi_init_d(prime_copy);
	cubi_init_d(max);
	cubi_init_d(zero);
	cubi_init_d(R);
	cubi_init_d(Q);
	id[0] = blockDim.x * blockIdx.x + threadIdx.x + 1;

	// Figure out starting prime
	cubi_div_d(root, numThreads, prime, R);
	prime[0]++;
	cubi_mult_d(prime, id, max);
	id[0]--;
	cubi_copy_d(prime, prime_copy);
	cubi_mult_d(prime_copy, id, prime);
	nextprime(prime);
	nextprime(max);

	// Loop through potential factors
	while (cubi_cmp_d(prime, max) <= 0 && done == 0) {
		// If prime divides N, add prime and q to factors, then break
		cubi_div_d(N, prime, Q, R);
		if (cubi_cmp_d(R, zero) == 0) {
			cubi_copy_d(prime, f1);
			cubi_copy_d(Q, f2);
			done = 1;
			for (int i = 0; i < SIZE; i++) {
				d_f1[i] = f1[i];
				d_f2[i] = f2[i];
			}
			break;
		} else {
			// Otherwise figure out next prime and continue
			nextprime(prime);
		}
	}

	free(id);
	free(prime);
	free(prime_copy);
	free(max);
	free(Q);
	free(R);
	free(zero);
	free(f1);
	free(f2);
	free(N);
	free(root);
	free(numThreads);
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
	if (argc < 3) {
		fprintf(stderr, "ERROR: You need to provide to primes to multiply\n");
		exit(1);
	}

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

	size_t size = SIZE * sizeof(unsigned int);


	// Calculate root(N), since I don't have a cubi function for that yet
	unsigned long long LN1 = atoi(argv[1]);
	unsigned long long LN2 = atoi(argv[2]);
	unsigned long long LN = LN1 * LN2;
	char rootStr[SIZE * 6];
	sprintf(rootStr, "%.0f", ceil(sqrt(LN)));

	// Allocate the host input values
	cubi root, N, N1, N2;
	root = (unsigned int*) malloc(size);
	N    = (unsigned int*) malloc(size);
	N1   = (unsigned int*) malloc(size);
	N2   = (unsigned int*) malloc(size);
	cubi_init_h(root);
	cubi_init_h(N);
	cubi_init_h(N1);
	cubi_init_h(N2);

	cubi_set_str_bin_h(N1, argv[1]);
	cubi_set_str_bin_h(N2, argv[2]);
	cubi_mult_h(N1, N2, N);
	cubi_set_str_bin_h(root, rootStr);

	cubi h_f1         = (unsigned int*) malloc(size);
	cubi h_f2         = (unsigned int*) malloc(size);
	cubi h_N          = (unsigned int*) malloc(size);
	cubi h_root       = (unsigned int*) malloc(size);
	cubi h_numThreads = (unsigned int*) malloc(size);
    int threadsPerBlock = 256;
    int blocksPerGrid = 7;
    int numThreads;
	numThreads = threadsPerBlock * blocksPerGrid;
	numThreads = threadsPerBlock * blocksPerGrid;
	for (int i = 0; i < SIZE; i++) {
		h_N[i] = N[i];
		h_root[i] = root[i];
	}
	h_numThreads[0] = numThreads;

/*    // Verify that allocations succeeded
    if (h_f1 == NULL || h_f1 == NULL || root == NULL || N == NULL)
    {
        fprintf(stderr, "Failed to allocate host values!\n");
        exit(EXIT_FAILURE);
    }*/

    // 1a. Allocate the device input vectors A & B
/*    double *d_f1 = NULL;
    err = cudaMalloc((void **)&d_f1, size);
    checkErr(err, "Failed to allocate device value d_f1");
    double *d_f2 = NULL;
    err = cudaMalloc((void **)&d_f2, size);
    checkErr(err, "Failed to allocate device value d_f2");*/
	//cubi d_f1;
	//cubi_init_cuda(&d_f1, SIZE);
	//cubi d_f2;
	//cubi_init_cuda(&d_f2, SIZE);
	cubi d_f1 = NULL;
    err = cudaMalloc((void **)&d_f1, size);
    checkErr(err, "Failed to allocate device value d_f1");
	cubi d_f2 = NULL;
    err = cudaMalloc((void **)&d_f2, size);
    checkErr(err, "Failed to allocate device value d_f2");
	cubi d_root = NULL;
    err = cudaMalloc((void **)&d_root, size);
    checkErr(err, "Failed to allocate device value d_root");
	cubi d_N = NULL;
    err = cudaMalloc((void **)&d_N, size);
    checkErr(err, "Failed to allocate device value d_N");
	cubi d_numThreads = NULL;
    err = cudaMalloc((void **)&d_numThreads, size);
    checkErr(err, "Failed to allocate device value d_numThreads");

	double wtime = -omp_get_wtime();
    // 2. Copy the host input vectors A and B in host memory 
    //     to the device input vectors in device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_f1, h_f1, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device value d_f1 from host to device");

    err = cudaMemcpy(d_f2, h_f2, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device value d_f2 from host to device");
    
	err = cudaMemcpy(d_root, h_root, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device value d_f1 from host to device");

    err = cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device value d_f2 from host to device");

    err = cudaMemcpy(d_numThreads, h_numThreads, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device value d_f2 from host to device");

    // 3. Launch the Vector Add CUDA Kernel
	vectorHypot<<<blocksPerGrid, threadsPerBlock>>>(d_root, d_N, d_f1, d_f2, d_numThreads);
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

	cubi f1, f2;
	f1 = (unsigned int*) malloc(size);
	f2 = (unsigned int*) malloc(size);
	cubi_init_h(f1);
	cubi_init_h(f2);
	for (int i = 0; i < SIZE; i++) {
		f1[i] = h_f1[i];
		f2[i] = h_f2[i];
	}
	printf("base 10?\t%f\t(%s x %s)\n", /*ceil(log10(N)),*/ wtime,
		cubi_get_str_bin_h(f1), cubi_get_str_bin_h(f2));

    // Free device global memory
    err = cudaFree(d_f1);
    checkErr(err, "Failed to free device vector A");

    err = cudaFree(d_f2);
    checkErr(err, "Failed to free device vector B");
	//cubi_free_cuda(&d_f1);
	//cubi_free_cuda(&d_f2);

    // Free host memory
    //cubi_free_h(&h_f1);
    //cubi_free_h(&h_f2);

    // Reset the device and exit
    err = cudaDeviceReset();
    checkErr(err, "Unable to reset device");

    return 0;
}

