/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Extended for use in CS 374 at Calvin College by Joel C. Adams.
 * Modified to measure timing and hypotenuse by Andrew Corum
 */

/**
 * Vector hypotenuse: C = sqrt(A * A + B * B).
 *
 * This sample is a very basic sample that implements element by element
 * vector hypotenuse. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some hypotenuses like error checking.
 */

#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include "Lock.h"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#define UNBURNT 0
#define SMOLDERING 1
#define BURNING 2
#define BURNT 3

#define true 1
#define false 0

typedef int boolean;

void seed_by_time(int offset) {
    time_t the_time;
    time(&the_time);
    srand((int)the_time+offset);
}

__device__
void initialize_forest(int forest_size, int * forest) {
    int i,j;

    for (i=0;i<forest_size;i++) {
        for (j=0;j<forest_size;j++) {
            forest[i+j*forest_size]=UNBURNT;
        }
    }
}

__device__
void light_tree(int forest_size, int * forest, int i, int j) {
    forest[i+j*forest_size]=SMOLDERING;
}

__device__
boolean forest_is_burning(int forest_size, int * forest) {
    int i,j;

    for (i=0; i<forest_size; i++) {
        for (j=0; j<forest_size; j++) {
            if (forest[i+j*forest_size]==SMOLDERING||forest[i+j*forest_size]==BURNING) {
                return true;
            }
        }
    }
    return false;
}

__device__
void forest_burns(int forest_size, int *forest, float prob_spread) {
    int i,j;
    extern boolean fire_spreads(float);

    //burning trees burn down, smoldering trees ignite
    for (i=0; i<forest_size; i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i+j*forest_size]==BURNING) forest[i+j*forest_size]=BURNT;
            if (forest[i+j*forest_size]==SMOLDERING) forest[i+j*forest_size]=BURNING;
        }
    }

    //unburnt trees catch fire
    for (i=0; i<forest_size; i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i+j*forest_size]==BURNING) {
                if (i!=0) { // North
                    if (fire_spreads(prob_spread)&&forest[i-1+j*forest_size]==UNBURNT) {
                        forest[i-1+j*forest_size]=SMOLDERING;
                    }
                }
                if (i!=forest_size-1) { //South
                    if (fire_spreads(prob_spread)&&forest[i+1+j*forest_size]==UNBURNT) {
                        forest[i+1+j*forest_size]=SMOLDERING;
                    }
                }
                if (j!=0) { // West
                    if (fire_spreads(prob_spread)&&forest[i+(j-1)*forest_size]==UNBURNT) {
                        forest[i+(j-1)*forest_size]=SMOLDERING;
                    }
                }
                if (j!=forest_size-1) { // East
                    if (fire_spreads(prob_spread)&&forest[i+(j+1)*forest_size]==UNBURNT) {
                        forest[i+(j+1)*forest_size]=SMOLDERING;
                    }
                }
            }
        }
    }
}

__device__
int burn_until_out(int forest_size, int * forest, float prob_spread,
    int start_i, int start_j) {
    int count;

    initialize_forest(forest_size,forest);
    light_tree(forest_size,forest,start_i,start_j);

    // burn until fire is gone
    count = 0;
    while(forest_is_burning(forest_size,forest)) {
        forest_burns(forest_size,forest,prob_spread);
        count++;
    }

    return count;
}

__device__
float get_percent_burned(int forest_size,int * forest) {
    int i,j;
    int total = forest_size*forest_size-1;
    int sum=0;

    // calculate pecrent burned
    for (i=0;i<forest_size;i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i+j*forest_size]==BURNT) {
                sum++;
            }
        }
    }

    // return percent burned;
    return ((float)(sum-1)/(float)total);
}

void checkErr(cudaError_t err, const char* msg) 
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s (error code %d: '%s')!\n", msg, err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void delete_forest(int forest_size, int * forest) {
    free(forest);
}

__device__
boolean fire_spreads(float prob_spread) {
    //if ((float)rand()/(float)RAND_MAX < prob_spread) 
    curandState_t state;
    curand_init(0, 0, 0, &state);
    int rand = curand(&state);
    if (0.5 < prob_spread) 
        return true;
    else
        return false;
}

void print_forest(int forest_size,int * forest) {
    int i,j;

    for (i=0;i<forest_size;i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i+j*forest_size]==BURNT) {
                printf(".");
            } else {
                printf("X");
            }
        }
        printf("\n");
    }
}

__device__
void lock(int* mutex) {
    while (atomicCAS(mutex, 0, 1) != 0) {}
}

__device__
void unlock(int* mutex) {
    atomicExch(mutex, 0);
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector hypotenuse of A and B into C. 
 * The 3 vectors have the same number of elements numElements.
 */
__global__
void startFire(
    float *A, float *B, int n_probs, int n_trials, int numThreads,
    int forest_size, int *forest, float * prob_spread, unsigned int* locks
)
{
	__shared__ int mutex;
	mutex = 0;
	__syncthreads();
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i_trial = id; i_trial < n_trials; i_trial += numThreads) {
        for (int i_prob = 0; i_prob < n_probs; i_prob++) {
			float b = (float) burn_until_out(forest_size, forest,
                prob_spread[i_prob], forest_size/2, forest_size/2);
            float a = get_percent_burned(forest_size, forest);
			atomicAdd(&(B[i_prob]), b);
			atomicAdd(&(A[i_prob]), a);
		}
    }
}

/**
 * Host main routine
 */
int main(int argc, char** argv)
{
    // initial conditions and variable definitions
    cudaError_t err = cudaSuccess;
    int forest_size=2;
    float * prob_spread;
    float * loc_prob_spread;
    float prob_min=0.0;
    float prob_max=1.0;
    float prob_step;
    int *forest;
    int *loc_forest;
    float * glo_percent_burned;    // %burn and iterations (global and local)
    float * loc_percent_burned;
    float * glo_num_iterations;
    float * loc_num_iterations;
    int n_trials=100;
    int i_prob;
    int n_probs=101;
    Lock myLock;
	unsigned int* locks;
	cudaMalloc((void**)&locks, sizeof(unsigned int)*n_trials);

    // check command line arguments

    if (argc > 1) {
        sscanf(argv[1],"%d",&forest_size);
    }
    if (argc > 2) {
        sscanf(argv[2],"%d",&n_trials);
    }
    if (argc > 3) {
        sscanf(argv[3],"%d",&n_probs);
    }

    // Print the vector length to be used, and compute its size
    size_t size = n_probs * sizeof(float);
    printf(
        "%dx%d forest, %d trials, %d probabilities.\n",
        forest_size, forest_size, n_trials, n_probs
    );
    
    // setup problem
    seed_by_time(0);
    forest = (int *) malloc(forest_size * forest_size * sizeof(int));
    prob_spread = (float *) malloc(size);
    glo_percent_burned = (float *) malloc(size);
    glo_num_iterations = (float *) malloc(size);

    // Verify that allocations succeeded
    if (glo_percent_burned == NULL || glo_num_iterations == NULL || prob_spread == NULL || forest == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize percent_burned to 0.0 for each prob
    prob_step = (prob_max-prob_min)/(float)(n_probs-1);
    for (i_prob = 0; i_prob < n_probs; i_prob++) {
        prob_spread[i_prob] = prob_min + (float)i_prob * prob_step;
        glo_percent_burned[i_prob]=0.0;
        glo_num_iterations[i_prob]=0.0;
    }
    
    // 1a. Allocate the device input vectors A & B
    loc_percent_burned = NULL;
    err = cudaMalloc((void **)&loc_percent_burned, size);
    checkErr(err, "Failed to allocate device vector loc_percent_burned");

    loc_num_iterations = NULL;
    err = cudaMalloc((void **)&loc_num_iterations, size);
    checkErr(err, "Failed to allocate device vector loc_num_iterations");

    loc_forest = NULL;
    err =cudaMalloc((void **)&loc_forest, forest_size * forest_size * sizeof(int));
    checkErr(err, "Failed to allocate device vector loc_forest");

    loc_prob_spread = NULL;
    err = cudaMalloc((void **)&loc_prob_spread, size);
    checkErr(err, "Failed to allocate device vector loc_prob_spread");
    
/*    // 1.b. Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    checkErr(err, "Failed to allocate device vector C");*/

    float wtimeA = -omp_get_wtime();
    // 2. Copy the host input vectors A and B in host memory 
    //     to the device input vectors in device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(loc_percent_burned, glo_percent_burned, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device vector loc_percent_burned from host to device");

    err = cudaMemcpy(loc_num_iterations, glo_num_iterations, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device vector loc_num_iterations from host to device");
    
    err = cudaMemcpy(loc_forest, forest, forest_size * forest_size * sizeof(int), cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device vector loc_forest from host to device");
    
    err = cudaMemcpy(loc_prob_spread, prob_spread, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device vector loc_prob_spread from host to device");
    wtimeA += omp_get_wtime();

    float wtimeB = -omp_get_wtime();
    // 3. Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(n_trials + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    startFire<<<blocksPerGrid, threadsPerBlock>>>(loc_percent_burned, loc_num_iterations, n_probs, n_trials,
        threadsPerBlock * blocksPerGrid, forest_size, loc_forest, loc_prob_spread, locks);
    err = cudaGetLastError();
    checkErr(err, "Failed to launch startFire kernel");
    wtimeB += omp_get_wtime();

    float wtimeC = -omp_get_wtime();
    // 4. Copy the device result vector in device memory
    //     to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(glo_percent_burned, loc_percent_burned, size, cudaMemcpyDeviceToHost);
    checkErr(err, "Failed to copy vector A from device to host");
    err = cudaMemcpy(glo_num_iterations, loc_num_iterations, size, cudaMemcpyDeviceToHost);
    checkErr(err, "Failed to copy vector B from device to host");
    wtimeC += omp_get_wtime();
    float wtime = wtimeA + wtimeB + wtimeC;

    printf("\nRuntimes:\n\tA: %f\n\tB: %f\n\tC: %f\n\tTotal: %f\n",
        wtimeA, wtimeB, wtimeC, wtime);

    // calculate averages and print output
    printf("Prob of spreading, Avg %% burned, Num of iterations\n");
    for ( i_prob = 0; i_prob < n_probs; i_prob++) {
        glo_percent_burned[i_prob]/=n_trials;
        glo_num_iterations[i_prob]/=n_trials;
        printf("%lf\t %lf\t %lf\n",prob_spread[i_prob],
            glo_percent_burned[i_prob],glo_num_iterations[i_prob]);
    }

    // Free device global memory
    err = cudaFree(loc_percent_burned);
    checkErr(err, "Failed to free device vector A");

    err = cudaFree(loc_num_iterations);
    checkErr(err, "Failed to free device vector B");

    float wtimeSeq = -omp_get_wtime();
/*    // repeat the computation sequentially
    for (int i = 0; i < numElements; ++i)
    {
       h_C[i] = sqrt(h_A[i] * h_A[i] + h_B[i] * h_B[i]);
    }
    wtimeSeq += omp_get_wtime();

    // verify again
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(sqrt(h_A[i] * h_A[i] + h_B[i] * h_B[i]) - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("\nNormal test PASSED\n");
    printf("\nSequential time: %f", wtimeSeq);*/

    // Free host memory
    delete_forest(forest_size,forest);
    free(prob_spread);
    free(glo_percent_burned);
    free(glo_num_iterations);

    // Reset the device and exit
    err = cudaDeviceReset();
    checkErr(err, "Unable to reset device");

    printf("\nDone\n");
    return 0;
}

