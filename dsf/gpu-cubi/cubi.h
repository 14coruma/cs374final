/* cubi.h contains the struct and method declarations for 
 * my CUDA Big Int data type
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
 * Constructor (host)
 *
 * @param: (cubi) a
 */
void cubi_init_h(cubi a) {
	for (int i = 0; i < SIZE; i++)
		a[i] = 0;
}

/**
 * Constructor (device)
 *
 * @param: (cubi) a
 */
__device__
void cubi_init_d(cubi a) {
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
	printf("STRUCT {\n\tsize: %d\n", SIZE);
	for (int i = 0; i < SIZE; i++) {
		printf("\t[%d]: %u\n", i, a[i]);
	}
	printf("}\n");
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
 * Bit shift cubi left (HOST)
 *
 * @param: Cubi
 */
void cubi_shift_left_h(cubi a) {
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
 * Bit shift cubi left (DEVICE)
 *
 * @param: Cubi
 */
__device__
void cubi_shift_left_d(cubi a) {
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
 * Copy constructor (HOST)
 * TODO: Work on different sized nums
 *
 * @param: (cubi) a
 * @param: (cubi) b
 */
void cubi_copy_h(cubi a, cubi b) {
	for (int i = 0; i < SIZE; i++) {
		b[i] = a[i];
	}
}

/**
 * Copy constructor (DEVICE)
 * TODO: Work on different sized nums
 *
 * @param: (cubi) a
 * @param: (cubi) b
 */
__device__
void cubi_copy_d(cubi a, cubi b) {
	for (int i = 0; i < SIZE; i++) {
		b[i] = a[i];
	}
}

/**
 * Set cubi value by string binary (HOST)
 *
 * @param: (cubi) a
 * @param: (cubi) str
 */
void cubi_set_str_bin_h(cubi a, char* str) {
	int length = strlen(str);
	if (length > SIZE * 32) {
		printf("\nERROR: cubi_set_str_bin(), string too long to store in cubi type.\n");
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
char* cubi_get_str_bin_h(cubi a) {
	int i;
	char* str = (char*) malloc(SIZE * 32 * sizeof(char));
	char* tempStr = (char*) malloc(32 * sizeof(char));

	// For each data element, add the number to str (through a tempStr)
	int leadZero = 1;
	for (i = SIZE - 1; i >= 0; i--) {
		if (a[i] == 0 && leadZero == 1) // Skip leading zeros
			continue;
		leadZero = 0;
		sprintf(tempStr, "%d", a[i]);
		strcat(str, tempStr);
	}

	// Copy space for retStr
	char* retStr = (char*) malloc(strlen(str));
	strcpy(retStr, str);

	// Remove leading un-wanted values
	for (int i = 0; i < 6; i++) {
		if (retStr[0] > '9' || retStr[0] < '0')
			retStr++;
	}
	free(str);
	free(tempStr);
	return retStr;
}
 
/**
 * Compare cubi numbers. (a == b) -> 0; (a < b) -> -1; (a > b) -> 1; (HOST)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * return: (int) {-1,0,1}
 */
int cubi_cmp_h(cubi a, cubi b) {
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
 * Compare cubi numbers. (a == b) -> 0; (a < b) -> -1; (a > b) -> 1; (DEVICE)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * return: (int) {-1,0,1}
 */
__device__
int cubi_cmp_d(cubi a, cubi b) {
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
 * Add two cubi numbers, saving the result in the third (a + b = c) (HOST)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
void cubi_add_h(cubi a, cubi b, cubi c) {
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
		printf("ERROR: cubi_add(), ran out of room in result variable\n");
		exit(1);
	}
}

/**
 * Add two cubi numbers, saving the result in the third (a + b = c) (DEVICE)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
__device__
void cubi_add_d(cubi a, cubi b, cubi c) {
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
		printf("ERROR: cubi_add(), ran out of room in result variable\n");
	}
}

/**
 * Subtract two cubi numbers, saving the result in the third (a - b = c) (HOST)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
void cubi_sub_h(cubi a, cubi b, cubi c) {
	if (cubi_cmp_h(a, b) < 0) {
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
 * Subtract two cubi numbers, saving the result in the third (a - b = c) (DEVICE)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
__device__
void cubi_sub_d(cubi a, cubi b, cubi c) {
	if (cubi_cmp_d(a, b) < 0) {
		printf("ERROR: cubi_sub(a, b, c), a must be greater than b\n");
		return;
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
 * Multiply two cubi numbers, saving the result in the third (a * b = c) (HOST)
 * TODO: Speedup with dynamic programming (different mult algorithm)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
void cubi_mult_h(cubi a, cubi b, cubi c) {
	cubi d, ccpy;
	d = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	ccpy = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	cubi_init_h(d);
	cubi_init_h(ccpy);

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
				cubi_copy_h(c, ccpy);
				cubi_add_h(ccpy, d, c);
				d[i+j] = 0;
			}
		}
	}

	// Carry should be empty. If not, we ran out of room
	if (carry > 0)
		printf("ERROR: cubi_add(), ran out of room in result variable");
	free(d);
	free(ccpy);
}

/**
 * Multiply two cubi numbers, saving the result in the third (a * b = c) (DEVICE)
 * TODO: Speedup with dynamic programming (different mult algorithm)
 *
 * @param: (cubi) a
 * @param: (cubi) b
 * @param: (cubi) c
 */
__device__
void cubi_mult_d(cubi a, cubi b, cubi c) {
	cubi d, ccpy;
	d = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	ccpy = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
	cubi_init_d(d);
	cubi_init_d(ccpy);

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
				cubi_copy_d(c, ccpy);
				cubi_add_d(ccpy, d, c);
				d[i+j] = 0;
			}
		}
	}

	// Carry should be empty. If not, we ran out of room
	if (carry > 0)
		printf("ERROR: cubi_add(), ran out of room in result variable");
	free(d);
	free(ccpy);
}

/**
 * Divide cubi numbers, storing the result in the third (N / D = Q) (HOST)
 * (DONE WITH BIT SHIFT)
 *
 * @param: (cubi) N
 * @param: (cubi) D
 * @param: (cubi) Q
 * @param: (cubi) R
 */
void cubi_div_h(cubi N, cubi D, cubi Q, cubi R) {
	// Clean out Q and R
	for (int i = 0; i < SIZE; i++) {
		Q[i] = 0;
		R[i] = 0;
	}
	cubi Rcpy;
	Rcpy = (unsigned int*) malloc(sizeof(unsigned int) * SIZE);

	// Check for divide by zero
	if (cubi_cmp_h(Q, D) == 0) {
		printf("ERROR: cubi_div(), DIVIDE BY ZERO\n");
		exit(1);
	}

	for (int i = SIZE * 32 - 1; i >= 0; i--) {
		cubi_shift_left_h(R);
		int tempN = (N[i/32] >> (i % 32)) & 1; // extract bit i/32, and put it at bit 0
		R[0] &= ~1;  // clear bit zero
		R[0] |= tempN; // Setbit 0 of R[0] to N[i/32]
		if (cubi_cmp_h(R, D) >= 0) {
			cubi_copy_h(R, Rcpy);
			cubi_sub_h(Rcpy, D, R);
			Q[i/32] |= (1 << (i % 32));
		}
	}
	free(Rcpy);
}

/**
 * Divide cubi numbers, storing the result in the third (N / D = Q) (DEVICE)
 * (DONE WITH BIT SHIFT)
 *
 * @param: (cubi) N
 * @param: (cubi) D
 * @param: (cubi) Q
 * @param: (cubi) R
 */
__device__
void cubi_div_d(cubi N, cubi D, cubi Q, cubi R) {
	// Clean out Q and R
	for (int i = 0; i < SIZE; i++) {
		Q[i] = 0;
		R[i] = 0;
	}
	cubi Rcpy;
	Rcpy = (unsigned int*) malloc(sizeof(unsigned int) * SIZE);

	// Check for divide by zero
	if (cubi_cmp_d(Q, D) == 0) {
		printf("ERROR: cubi_div(), DIVIDE BY ZERO\n");
	}

	for (int i = SIZE * 32 - 1; i >= 0; i--) {
		cubi_shift_left_d(R);
		int tempN = (N[i/32] >> (i % 32)) & 1; // extract bit i/32, and put it at bit 0
		R[0] &= ~1;  // clear bit zero
		R[0] |= tempN; // Setbit 0 of R[0] to N[i/32]
		if (cubi_cmp_d(R, D) >= 0) {
			cubi_copy_d(R, Rcpy);
			cubi_sub_d(Rcpy, D, R);
			Q[i/32] |= (1 << (i % 32));
		}
	}
	free(Rcpy);
}
