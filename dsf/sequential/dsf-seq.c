/* dsf.c is a prime factorization program using Direct Search Factorization
 *
 * Andrew Corum, Dec 2017
 * Usage: dsf [num] [num] [table format?]
 */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <mpi.h>

#define MAX_FACTORS 40
#define true 1
#define false 0
typedef int boolean;

int main(int argc, char ** argv) {
	// Create array to hold factors
	mpz_t factors[MAX_FACTORS];

	// Create mpz_t variables
	mpz_t prime, root, r, q, N, N1, N2, zero;
	int digits = 0;

	// Check for command line args
	if (argc >= 3) {
		// Multiply N = N1 x N2
		mpz_init(N);
		mpz_init_set_str(N1, argv[1], 10);
		mpz_init_set_str(N2, argv[2], 10);
		mpz_mul(N, N1, N2);
	} else if (argc == 2) {
		// Assign N = argv[1]
		mpz_init_set_str(N, argv[1], 10);
	} else {
		printf("ERROR: Need a number to factorize (or product of numbers).\n");
		return 1;
	}


	// Init/assign mpz variables
	mpz_init_set_str(prime, "2", 10);
	mpz_init_set_str(zero, "0", 10);
	mpz_init(root);
	mpz_init(r);
	mpz_init(q);
	digits = mpz_sizeinbase(N, 10);
	for (int i = 0; i < MAX_FACTORS; i++) {
		mpz_init(factors[i]);
	}

	// Calculate floor(sqrt(N)) as an upper bound for factors
	mpz_sqrt(root, N);

	// Start timer
	double time = -MPI_Wtime();

	int i = 0;
	// Loop through potential factors
	while (mpz_cmp(prime, root) <= 0 && i < MAX_FACTORS) {
		// If N is prime, we're done
		if (mpz_probab_prime_p(N, 15) > 0) {
			mpz_set(factors[i], N);
			i++;
			break;
		}
		// Try to divide N by prime
		mpz_fdiv_qr(q, r, N, prime);

		// If prime divides N, add prime to list of factors and continue
		if (mpz_cmp(r, zero) == 0) {
			mpz_set(factors[i], prime);
			mpz_set_str(prime, "2", 10);
			mpz_set(N, q);
			i++;
		} else {
			mpz_nextprime(prime, prime); // otherwise go to next prime
		}
	}

	// Stop timer
	time += MPI_Wtime();

	// Print results
	if (argc != 4) {
		printf("%d factors found in %f seconds.\n", i, time);
		
		if (argc == 3) {
			printf("Prime factors of %s x %s:\n", argv[1], argv[2]);
		} else if (argc == 2) {
			printf("Prime factors of %s:\n", argv[1]);
		}
		for (--i; i >= 0; i--) {
			printf("%s ", mpz_get_str(NULL, 10, factors[i]));
		}
	} else {
		printf("%d\t%f\t(%s x %s)\t(%s x %s)", digits, time, argv[1], argv[2],
			mpz_get_str(NULL, 10, factors[0]), mpz_get_str(NULL, 10, factors[1])
		);
	}
	printf("\n");
	return 0;
}
