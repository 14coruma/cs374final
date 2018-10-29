# cs374final
Final project for CS 374 HPC class

`/FIRE` Contains an attempt at parallelizing a Monte Carlo forest fire simulation. However, this project was not successful due to deadlock between GPU warps.

`/dsf` Contains a parallelized GPU implementation of Direct Search Factorization, which is capable of factoring large primes (on the order of 10^18) fairly quickly.

This directory also contains my own Cuda Big-Int library, called CUBI. Since CUDA does not have an authoritative library for multi-precision arithmatic, I implemented my own for this project.
