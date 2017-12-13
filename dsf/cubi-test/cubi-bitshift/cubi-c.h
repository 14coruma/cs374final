/* cubi-c.h contains the struct and method declarations for 
 * my CUDA Big Int data type (for testing with c)
 *
 * Andrew Corum, Dec 2017
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define SIZE 12
#define INT_MAX 4294967295
#define LEFT_ONE (1 << ((sizeof(unsigned int) * 8) - 1))

typedef unsigned int* cubi;

/**
 * Constructor
 *
 * @param: (cubi) a
 */
void cubi_init(cubi a) {
	for (int i = 0; i < SIZE; i++)
		a[i] = 0;
}

/**
 * Destructor
 *
 * @param: (cubi) a
 */
void cubi_free(cubi a) {
	free(a);
}

/**
 * Get Size of cubi
 *
 * @param: (cubi) a
 * return: (int) size
 */
int cubi_size(cubi a) {
	return SIZE;
}

/**
 * Debug dump of cubi contents
 *
 * @param: (cubi) a
 */
void cubi_dump(cubi a) {
	fprintf(stderr, "STRUCT {\n\tsize: %d\n", SIZE);
	for (int i = 0; i < SIZE; i++) {
		fprintf(stderr, "\t[%d]: %u\n", i, a[i]);
	}
	fprintf(stderr, "}\n");
}

/**
 * Bit shift cubi right
 *
 * @param: Cubi
 */
void cubi_shift_right(cubi a) {
	int bits1, bits2;
	bits1 = bits2 = 0;
	for (int i = SIZE - 1; i >= 0; i--) {
		bits1 = a[i] & 1;
		a[i] >>= 1;
		//a[i] |= (bits2 * ( 1 << ((sizeof(int) - 1)) * 8));
		if (bits2 == 1) a[i] += 500000;
		bits2 = bits1;
	}
}

/**
 * Bit shift cubi left
 *
 * @param: Cubi
 */
void cubi_shift_left(cubi a) {
	unsigned int bits1, bits2;
	bits1 = bits2 = 0;
	for (int i = 0; i < SIZE; i++) {
		bits1 = a[i] & (1 << ((sizeof(unsigned int)) * 8 - 1));
		a[i] <<= 1;
		if (bits2) {
			a[i] += 1;
		}
		bits2 = bits1;
	}
}

/**
 * Copy constructor
 * TODO: Work on different sized nums
 *
 * @param: (cubi) a
 * @param: (cubi) b
 */
void cubi_copy(cubi a, cubi b) {
	for (int i = 0; i < SIZE; i++) {
		b[i] = a[i];
	}
}

/**
 * Set cubi value by string
 *
 * @param: (cubi) a
 * @param: (char*) str
 */
/*void cubi_set_str(cubi a, char* str) {
	int length = strlen(str);
	if (length > SIZE * 6)
		fprintf(stderr, "\nERROR: cubi_set_str(), string too long to store in cubi type.\n");
	else {
		// Empty out cubi
		for (int i = 0; i < SIZE; i++)
			a[i] = 0;

		// Starting at the right end of string, add to from left end of array
		unsigned long int R = 0;
		int i = 0; int mod = 0;
		while (length > 0) {
			R += ((str[length-1] - '0') * (unsigned long int) pow(10, mod));
			fprintf(stderr, "i: %d\tR: %lu\t",i, R);
			if (R > INT_MAX) {
				int carry = 0;
				while (R > INT_MAX) {
					R -= INT_MAX;
					carry++;
				}
				a[i] = R;
				R = carry;
				i++;
				mod = 0;
			} else {
				a[i] = R;
				mod++;
			}
			length--;
		}
			a[i] = R;
			fprintf(stderr, "i: %d\tR: %lu\t",i, R);
	}
}*/

/**
 * Set cubi value by string binary
 *
 * @param: (cubi) a
 * @param: (cubi) str
 */
void cubi_set_str_bin(cubi a, char* str) {
	int length = strlen(str);
	if (length > SIZE * 32) {
		fprintf(stderr, "\nERROR: cubi_set_str_bin(), string too long to store in cubi type.\n");
		exit(1);
	}

	// Empty out cubi
	for (int i = 0; i < SIZE; i++)
		a[i] = 0;

	for (int i = 0; i < length; i++) {
		a[i/32] += (str[length-i-1] - '0') * (unsigned int) pow(2, i % 32);
	}
}

/**
 * Get string representation of cubi data
 * TODO: Fix - Prints incorrectly when there are leading zeros in the data[] int
 *       (since they don't show up in the sprintf)
 *
 * @param: (cubi) a
 * return: (char*) retStr
 */
char* cubi_get_str_bin(cubi a) {
	int i;
	char* str = (char*) malloc(SIZE * 32 * sizeof(char));
	char* tempStr = (char*) malloc(32 * sizeof(char));
	char* bit = (char*) malloc(32 * sizeof(char));

	// For each data element, add the number to str (through a tempStr)
	int leadZero = 1;
	for (i = SIZE - 1; i >= 0; i--) {
		if (a[i] == 0 && leadZero == 1) // Skip leading zeros
			continue;
		leadZero = 0;
		for (int j = 0; j < 32; j++) {
			bit[j] = ((a[i] & (1 << j)) >> j) + '0'; // Set bit[j] to the binary character in a[i] at j
		}
		strcat(str, bit);
	}

	// Copy space for retStr
	char* retStr = (char*) malloc(strlen(str));
	strcpy(retStr, str);

	// Remove leading un-wanted values
	for (int i = 0; i < 6; i++) {
		if (retStr[0] > '1' || retStr[0] < '0')
			retStr++;
	}
	free(str);
	free(tempStr);
	free(bit);
	return retStr;
}

/**
 * Get order of magnitude of cubi (with respect to 10^6);
 *
 * @param: (cubi) a
 * return: (int) order of magnitude
 */
int cubi_magnitude(cubi a) {
	int i;
	for (i = SIZE - 1; i >= 0; i--) {
		if (a[i] != 0)
			break;
	}
	if (i < 0) i = 0;
	return i;
}
 
/**
 * Compare cubi numbers. (a == b) -> 0; (a < b) -> -1; (a > b) -> 1;
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * return: (int) {-1,0,1}
 */
int cubi_cmp(cubi a, cubi b) {
	for (int i = SIZE - 1; i >= 0; i--) {
		if (a[i] > b[i])
			return 1;
		else if (a[i] < b[i])
			return -1;
		else
			continue;
	}
	return 0;
}

/**
 * Add two cubi numbers, saving the result in the third (a + b = c)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
void cubi_add(cubi a, cubi b, cubi c) {
	// Clean out c
	for (int i = 0; i < SIZE; i++)
		c[i] = 0;

	unsigned int carry = 0;
	for (int i = 0; i < SIZE; i++) {
		c[i] = a[i] + b[i] + carry;
		carry = 0;
		if (c[i] < a[i] || c[i] < b[i]) {
			carry = 1;
			c[i] -= INT_MAX + 1;
		}
	}

	// Carry should be empty. If not, we ran out of room
	if (carry > 0) {
		fprintf(stderr, "ERROR: cubi_add(), ran out of room in result variable\n");
		exit(1);
	}
}

/**
 * Subtract two cubi numbers, saving the result in the third (a - b = c)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
void cubi_sub(cubi a, cubi b, cubi c) {
	if (cubi_cmp(a, b) < 0) {
		printf("ERROR: cubi_sub(a, b, c), a must be greater than b\n");
		exit(1);
	}

	// Clean out c
	for (int i = 0; i < SIZE; i++)
		c[i] = 0;

	for (int i = 0; i < SIZE; i++) {
		if (b[i] > a[i]) {
			c[i+1] -= 1;
		}
		c[i] += a[i] - b[i];
	}
}

/**
 * Multiply two cubi numbers, saving the result in the third (a * b = c)
 * TODO: Speedup with dynamic programming (different mult algorithm)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
void cubi_mult(cubi a, cubi b, cubi c) {
	cubi d, ccpy;
	d = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	ccpy = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	cubi_init(d);
	cubi_init(ccpy);

	// Clean out c
	for (int i = 0; i < SIZE; i++)
		c[i] = 0;

	unsigned long carry = 0;
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			if (j+i < SIZE) {
				unsigned long ul = (unsigned long) a[i] * (unsigned long) b[j] + (unsigned long) carry;
				carry = (unsigned int) (ul / (INT_MAX+1));
				d[i+j] = (unsigned int) (ul - carry * (INT_MAX+1));
				cubi_copy(c, ccpy);
				cubi_add(ccpy, d, c);
				d[i+j] = 0;
			}
		}
	}

	// Carry should be empty. If not, we ran out of room
	if (carry > 0)
		fprintf(stderr, "ERROR: cubi_add(), ran out of room in result variable");
	cubi_free(d);
	cubi_free(ccpy);
}

/**
 * Divide cubi numbers, storing the result in the third (a / b = c)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
/*void cubi_div(cubi a, cubi b, cubi c) {
	int sa = SIZE;
	int sb = SIZE;
	int sc = SIZE;

	if (sa != sb || sa != sc) {
		printf("ERROR: cubi_div(a, b, c) requires that size of a,b,c are equal\n");
		exit(1);
	}

	// Clean out c
	for (int i = 0; i < sc; i++)
		c[i] = 0;

	cubi zero, R, Rcpy, BxRES, RES, OVER;
	cubi_init(zero);
	if (cubi_cmp(zero, b) == 0) {
		fprintf(stderr, "ERROR: cubi_div(), divide by zero.\n");
		exit(1);
	}

	cubi_init(BxRES);
	cubi_init(R);
	cubi_init(Rcpy);
	cubi_init(RES);
	cubi_init(OVER);
	cubi_copy(Rcpy, R);
	int magB = cubi_magnitude(b);
	int carry = 0; int overshot = 0;
	while (cubi_cmp(R, b) >= 0) {
		int magR = cubi_magnitude(R);

		// Inital guess of c[magR - magB]
		long int res = R[magR] / b[magB];
		if (res == 0) {
			res = (R[magR] * 1000000) / b[magB];
			carry = 1;
		}
		if (res <= 0) { res *= -1; res++; }
		c[magR - magB - carry] += res - overshot;
		RES[magR - magB - carry] = res - overshot;
		cubi_mult(b, RES, BxRES);
		cubi_copy(R, Rcpy);
		if (cubi_cmp(R, BxRES) < 0) { // Then we overshot. Add remove from c
			c[magR - magB - carry] -= res - overshot;
//			overshot++;
			cubi_sub(BxRES, R, OVER);
			int magOver = cubi_magnitude(OVER);
			overshot += OVER[magOver] / b[magB] + 2; // add estimate for overshoot
		} else {
			cubi_sub(Rcpy, BxRES, R);
			overshot = 0;
		}
		RES[magR - magB - carry] = 0;
		carry = 0;
	}

	cubi_free(zero);
	cubi_free(R);
	cubi_free(Rcpy);
	cubi_free(RES);
	cubi_free(BxRES);
	cubi_free(OVER);
}*/

/**
 * Divide cubi numbers, storing the result in the third (N / D = Q)
 * (DONE WITH BIT SHIFT)
 *
 * @param: (cubi) N
 * @param: (cubi) D
 * @param: (cubi) Q
 * @param: (cubi) R
 */
void cubi_div(cubi N, cubi D, cubi Q, cubi R) {
	// Clean out Q and R
	for (int i = 0; i < SIZE; i++) {
		Q[i] = 0;
		R[i] = 0;
	}
	cubi Rcpy;
	Rcpy = (unsigned int*) malloc(sizeof(unsigned int) * SIZE);

	// Check for divide by zero
	if (cubi_cmp(Q, D) == 0) {
		printf("ERROR: cubi_div(), DIVIDE BY ZERO\n");
		exit(1);
	}

	for (int i = SIZE * 32 - 1; i >= 0; i--) {
		cubi_shift_left(R);
		int tempN = (N[i/32] >> (i % 32)) & 1; // extract bit i/32, and put it at bit 0
		R[0] &= ~1;  // clear bit zero
		R[0] |= tempN; // Setbit 0 of R[0] to N[i/32]
		if (cubi_cmp(R, D) >= 0) {
			cubi_copy(R, Rcpy);
			cubi_sub(Rcpy, D, R);
			Q[i/32] |= (1 << (i % 32));
		}
	}
}

/**
 * Calculate modulus of a cubi number: a (mod b) = c
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
void cubi_mod(cubi a, cubi b, cubi c) {
	int sa = SIZE;
	int sb = SIZE;
	int sc = SIZE;

	if (sa != sb || sa != sc) {
		printf("ERROR: cubi_mod(a, b, c) requires that size of a,b,c are equal\n");
		exit(1);
	}

	// Clean out c
	for (int i = 0; i < sc; i++)
		c[i] = 0;

	cubi Q, D;
	cubi_init(D);
	cubi_init(Q);
	//cubi_div(a, b, Q);
	cubi_mult(b, Q, D);
	cubi_sub(a, D, c);

	cubi_free(Q);
	cubi_free(D);
}
