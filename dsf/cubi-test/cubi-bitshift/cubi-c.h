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

typedef int* cubi;

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
		fprintf(stderr, "\t[%d]: %d\n", i, a[i]);
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
	//	bits = a[i] & (1 << (sizeof(int) - 1)*8); // for shift left, oops
		bits1 = a[i] & 1;
		a[i] >>= 1;
		//a[i] |= (bits2 * ( 1 << ((sizeof(int) - 1)) * 8));
		if (bits2 == 1) a[i] += 500000;
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
void cubi_set_str(cubi a, char* str) {
	int length = strlen(str);
	if (length > SIZE * 6)
		fprintf(stderr, "\nERROR: cubi_set_str(), string too long to store in cubi type.\n");
	else {
		// Empty out cubi
		for (int i = 0; i < SIZE; i++)
			a[i] = 0;

		// Starting at the right end of string, add to from left end of array
		for (int i = 0; i < length; i++) {
			a[i/6] += (str[length-i-1] - '0') * (int) pow(10, i % 6);
		}
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
char* cubi_get_str(cubi a) {
	int i;
	int size = SIZE;
	char* str = (char*) malloc(size * 6 * sizeof(char));
	char* tempStr = (char*) malloc(6 * sizeof(char));

	// For each data element, add the number to str (through a tempStr)
	int leadZero = 1;
	for (i = size - 1; i >= 0; i--) {
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

	int carry = 0;
	for (int i = 0; i < SIZE; i++) {
		c[i] = a[i] + b[i] + carry;
		carry = c[i] / 1000000;
		c[i] -= carry * 1000000;
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
	int sa = SIZE;
	int sb = SIZE;
	int sc = SIZE;
	if (sa != sb || sa != sc) {
		printf("ERROR: cubi_sub(a, b, c) requires that size of a,b,c are equal\n");
		exit(1);
	}
	if (cubi_cmp(a, b) < 0) {
		printf("ERROR: cubi_sub(a, b, c), a must be greater than b\n");
		exit(1);
	}

	// Clean out c
	for (int i = 0; i < SIZE; i++)
		c[i] = 0;

	int carry = 0;
	for (int i = 0; i < sa; i++) {
		if (b[i] > a[i]) {
			carry = 1000000;
			c[i+1] -= 1;
		}
		c[i] += a[i] - b[i] + carry;
		carry = 0;
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
	d = (int*) malloc(SIZE * sizeof(int));
	ccpy = (int*) malloc(SIZE * sizeof(int));
	cubi_init(d);
	cubi_init(ccpy);

	// Clean out c
	for (int i = 0; i < SIZE; i++)
		c[i] = 0;

	int carry = 0;
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			if (j+i < SIZE) {
				unsigned long ul = (unsigned long) a[i] * (unsigned long) b[j] + (unsigned long) carry;
				carry = (int) (ul / 1000000);
				d[i+j] = (int) (ul - carry * 1000000);
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
 */
/*void cubi_div(cubi N, cubi D, cubi Q) {
	// Clean out Q
	for (int i = 0; i < SIZE; i++)
		Q[i] = 0;

	// Check for divide by zero
	if ()
}*/

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
	cubi_div(a, b, Q);
	cubi_mult(b, Q, D);
	cubi_sub(a, D, c);

	cubi_free(Q);
	cubi_free(D);
}
