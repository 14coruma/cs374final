/* dsf.c is a prime factorization program using Direct Search Factorizatoin
 *
 * Andrew Corum, Dec 2017
 * Usage: 
 */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

#define true 1
#define false 0
typedef int boolean;

int main(int argc, char ** argv) {
	mpz_t r, n, d;
	if (argc < 3) {
		printf("Need two numbers to add\n");
		return 1;
	}
	mpz_init_set_str(n, argv[1], 10);
	mpz_init_set_str(d, argv[2], 10);
	mpz_init(r);
	mpz_fdiv_r(r, n, d);

	printf("%s / %s => r = %s\n", argv[1], argv[2], mpz_get_str(NULL, 10, r));
	return 0;
}
