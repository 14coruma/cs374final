/* cubi.h contains the struct and method declarations for 
 * my CUDA Big Int data type
 *
 * Andrew Corum, Dec 2017
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct cubi_z {
	int *data; // The number (left is least significant)
	int size;  // Number of elements in data[]
};

typedef struct cubi_z cubi;

/**
 * Constructor (for the cuda device)
 *
 * @param: (cubi*) a
 * @param: (int) size
 */
void cubi_init_cuda(cubi* a, int size) {
	cudaMalloc((void **)a, (size+1) * sizeof(int));
	a->data = (int*) malloc(size * sizeof(int));
	for (int i = 0; i < size; i++) {
		a->data[i] = 0;
	}
	a->size = size;
}

/**
 * Constructor (device)
 *
 * @param: (cubi*) a
 * @param: (int) size
 */
__device__
void cubi_init_d(cubi* a, int size) {
	a->data = (int*) malloc(size * sizeof(int));
	for (int i = 0; i < size; i++)
		a->data[i] = 0;
	a->size = size;
}

/**
 * Constructor (for the host)
 *
 * @param: (cubi*) a
 * @param: (int) size
 */
void cubi_init_h(cubi* a, int size) {
	a->data = (int*) malloc(size * sizeof(int));
	for (int i = 0; i < size; i++)
		a->data[i] = 0;
	a->size = size;
}

/**
 * Destructor (cuda)
 *
 * @param: (cubi*) a
 */
void cubi_free_cuda(cubi* a) {
	cudaFree(a->data);
}

/**
 * Destructor (host)
 *
 * @param: (cubi*) a
 */
void cubi_free_h(cubi* a) {
	free(a->data);
}

/**
 * Destructor (device)
 *
 * @param: (cubi*) a
 */
__device__
void cubi_free_d(cubi* a) {
	free(a->data);
}

/**
 * Get Size of cubi
 *
 * @param: (cubi*) a
 * return: (int) size
 */
__device__
int cubi_size(cubi* a) {
	return a->size;
}

/**
 * Debug dump of cubi contents
 *
 * @param: (cubi*) a
 */
__device__
void cubi_dump(cubi* a) {
	printf("STRUCT {\n\tsize: %d\n", a->size);
	for (int i = 0; i < cubi_size(a); i++) {
		printf("\t[%d]: %d\n", i, a->data[i]);
	}
	printf("}\n");
}

/**
 * Copy constructor (ON HOST)
 * TODO: Work on different sized nums
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 */
void cubi_copy_h(cubi* a, cubi* b) {
	if (a->size != b->size) {
		printf("ERROR: cubi_copy(), Cubis must be of the same size.\n");
		return;
	}
	for (int i = 0; i < a->size; i++) {
		b->data[i] = a->data[i];
	}
}

/**
 * Copy constructor
 * TODO: Work on different sized nums
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 */
__device__
void cubi_copy(cubi* a, cubi* b) {
	if (a->size != b->size) {
		printf("ERROR: cubi_copy(), Cubis must be of the same size.\n");
		return;
	}
	for (int i = 0; i < a->size; i++) {
		b->data[i] = a->data[i];
	}
}

/**
 * Calculates the power of an int (that can be used on the CUDA device) (ON HOST)
 * 
 * @param: (int) base
 * @param: (int) power
 * return: (int) base ^ power
 */
int power_h(int base, int power) {
	int result = 1;
	for (int i = 0; i < power; i++) {
		result *= base;
	}
	return result;
}

/**
 * Calculates the power of an int (that can be used on the CUDA device)
 * 
 * @param: (int) base
 * @param: (int) power
 * return: (int) base ^ power
 */
__device__
int power(int base, int power) {
	int result = 1;
	for (int i = 0; i < power; i++) {
		result *= base;
	}
	return result;
}

/**
 * Set cubi value by string (for device)
 *
 * @param: (cubi*) a
 * @param: (char*) str
 * @param: (int) length, the string length
 */
__device__
void cubi_set_str(cubi* a, char* str, int length) {
	if (length > cubi_size(a) * 6)
		printf("\nERROR: cubi_set_str(), string too long to store in cubi type.\n");
	else {
		// Empty out cubi
		for (int i = 0; i < cubi_size(a); i++)
			a->data[i] = 0;

		// Starting at the right end of string, add to data[] from left end of array
		for (int i = 0; i < length; i++) {
			a->data[i/6] += (str[length-i-1] - '0') * (int) power(10, i % 6);
		}
	}
}

/**
 * Set cubi value by string (for host)
 *
 * @param: (cubi*) a
 * @param: (char*) str
 */
void cubi_set_str_h(cubi* a, char* str) {
	int length = strlen(str);
	if (length > a->size * 6)
		printf("\nERROR: cubi_set_str(), string too long to store in cubi type.\n");
	else {
		// Empty out cubi
		for (int i = 0; i < a->size; i++)
			a->data[i] = 0;

		// Starting at the right end of string, add to data[] from left end of array
		for (int i = 0; i < length; i++) {
			a->data[i/6] += (str[length-i-1] - '0') * (int) power_h(10, i % 6);
		}
	}
}

/**
 * Get string representation of cubi data
 * TODO: Fix - Prints incorrectly when there are leading zeros in the data[] int
 *       (since they don't show up in the sprintf)
 *
 * @param: (cubi*) a
 * return: (char*) retStr
 */
char* cubi_get_str_h(cubi* a) {
	int i;
	int size = a->size;
	char* str = (char*) malloc(size * 6 * sizeof(char));
	char* tempStr = (char*) malloc(6 * sizeof(char));

	// For each data element, add the number to str (through a tempStr)
	int leadZero = 1;
	for (i = size - 1; i >= 0; i--) {
		if (a->data[i] == 0 && leadZero == 1) // Skip leading zeros
			continue;
		leadZero = 0;
		sprintf(tempStr, "%d", a->data[i]);
		strcat(str, tempStr);
	}

	// Copy space for retStr
	char* retStr = (char*) malloc(size * 6);
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
 * @param: (cubi*) a
 * return: (int) order of magnitude
 */
__device__
int cubi_magnitude(cubi* a) {
	int i;
	for (i = a->size - 1; i >= 0; i--) {
		if (a->data[i] != 0)
			break;
	}
	if (i < 0) i = 0;
	return i;
}
 
/**
 * Compare cubi numbers. (a == b) -> 0; (a < b) -> -1; (a > b) -> 1;
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 * return: (int) {-1,0,1}
 */
__device__
int cubi_cmp(cubi* a, cubi* b) {
	int sa = cubi_size(a);
	int sb = cubi_size(b);
	int smin = (sa < sb) ? sa : sb; // MIN(sa, sb);
	int smax = (sa < sb) ? sb : sa; // MAX(sa, sb);
	for (int i = smax - 1; i >= 0; i--) {
		if (i < smin - 1) {
			if (a->data[i] > b->data[i])
				return 1;
			else if (a->data[i] < b->data[i])
				return -1;
			else
				continue;
		}
		if (i > sa - 1) {
			if (b->data[i] > 0) return -1;
		} else if (i > sb - 1) {
			if (a->data[i] > 0) return 1;
		}
	}
	return 0;
}

/**
 * Add two cubi numbers, saving the result in the third (a + b = c) (ON HOST)
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 * @param: (cubi*) c
 */
void cubi_add_h(cubi* a, cubi* b, cubi* c) {
	int sa = a->size;
	int sb = b->size;
	int size = (sa < sb) ? sa : sb; // MIN(sa, sb);

	// Make sure c is big enough
	if (c->size < size) {
		printf("ERROR: cubi_add(), result cubi is too small to store sum\n");
		return;
	}

	// Clean out c
	for (int i = 0; i < c->size; i++)
		c->data[i] = 0;

	int carry = 0;
	for (int i = 0; i < size; i++) {
		c->data[i] = a->data[i] + b->data[i] + carry;
		carry = c->data[i] / 1000000;
		c->data[i] -= carry * 1000000;
	}

	// Carry should be empty. If not, we ran out of room
	if (carry > 0) {
		printf("ERROR: cubi_add(), ran out of room in result variable\n");
		return;
	}
}

/**
 * Add two cubi numbers, saving the result in the third (a + b = c)
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 * @param: (cubi*) c
 */
__device__
void cubi_add(cubi* a, cubi* b, cubi* c) {
	int sa = cubi_size(a);
	int sb = cubi_size(b);
	int size = (sa < sb) ? sa : sb; // MIN(sa, sb);

	// Make sure c is big enough
	if (cubi_size(c) < size) {
		printf("ERROR: cubi_add(), result cubi is too small to store sum\n");
		return;
	}

	// Clean out c
	for (int i = 0; i < cubi_size(c); i++)
		c->data[i] = 0;

	int carry = 0;
	for (int i = 0; i < size; i++) {
		c->data[i] = a->data[i] + b->data[i] + carry;
		carry = c->data[i] / 1000000;
		c->data[i] -= carry * 1000000;
	}

	// Carry should be empty. If not, we ran out of room
	if (carry > 0) {
		printf("ERROR: cubi_add(), ran out of room in result variable\n");
		return;
	}
}

/**
 * Subtract two cubi numbers, saving the result in the third (a - b = c)
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 * @param: (cubi*) c
 */
__device__
void cubi_sub(cubi* a, cubi* b, cubi* c) {
	int sa = cubi_size(a);
	int sb = cubi_size(b);
	int sc = cubi_size(c);
	if (sa != sb || sa != sc) {
		printf("ERROR: cubi_sub(a, b, c) requires that size of a,b,c are equal\n");
		return;
	}
	if (cubi_cmp(a, b) < 0) {
		printf("ERROR: cubi_sub(a, b, c), a must be greater than b\n");
		return;
	}

	// Clean out c
	for (int i = 0; i < cubi_size(c); i++)
		c->data[i] = 0;

	int carry = 0;
	for (int i = 0; i < sa; i++) {
		if (b->data[i] > a->data[i]) {
			carry = 1000000;
			c->data[i+1] -= 1;
		}
		c->data[i] += a->data[i] - b->data[i] + carry;
		carry = 0;
	}
}

/**
 * Multiply two cubi numbers, saving the result in the third (a * b = c) (ON HOST)
 * TODO: Speedup with dynamic programming (different mult algorithm)
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 * @param: (cubi*) c
 */
void cubi_mult_h(cubi* a, cubi* b, cubi* c) {
	int sa = a->size;
	int sb = b->size;
	int sc = c->size;

	cubi d, ccpy;
	cubi_init_h(&d, sa + sb);
	cubi_init_h(&ccpy, sc);

	// Clean out c
	for (int i = 0; i < sc; i++)
		c->data[i] = 0;

	int carry = 0;
	for (int i = 0; i < sa; i++) {
		for (int j = 0; j < sb; j++) {
			if (j+i < sc) {
				unsigned long ul = (unsigned long) a->data[i] * (unsigned long) b->data[j] + (unsigned long) carry;
				carry = (int) (ul / 1000000);
				d.data[i+j] = (int) (ul - carry * 1000000);
				cubi_copy_h(c, &ccpy);
				cubi_add_h(&ccpy, &d, c);
				d.data[i+j] = 0;
			}
		}
	}

	// Carry should be empty. If not, we ran out of room
	if (carry > 0)
		printf("ERROR: cubi_add(), ran out of room in result variable");
	cubi_free_h(&d);
	cubi_free_h(&ccpy);
}

/**
 * Multiply two cubi numbers, saving the result in the third (a * b = c)
 * TODO: Speedup with dynamic programming (different mult algorithm)
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 * @param: (cubi*) c
 */
__device__
void cubi_mult(cubi* a, cubi* b, cubi* c) {
	int sa = cubi_size(a);
	int sb = cubi_size(b);
	int sc = cubi_size(c);

	cubi d, ccpy;
	cubi_init_d(&d, sa + sb);
	cubi_init_d(&ccpy, sc);

	// Clean out c
	for (int i = 0; i < sc; i++)
		c->data[i] = 0;

	int carry = 0;
	for (int i = 0; i < sa; i++) {
		for (int j = 0; j < sb; j++) {
			if (j+i < sc) {
				unsigned long ul = (unsigned long) a->data[i] * (unsigned long) b->data[j] + (unsigned long) carry;
				carry = (int) (ul / 1000000);
				d.data[i+j] = (int) (ul - carry * 1000000);
				cubi_copy(c, &ccpy);
				cubi_add(&ccpy, &d, c);
				d.data[i+j] = 0;
			}
		}
	}

	// Carry should be empty. If not, we ran out of room
	if (carry > 0)
		printf("ERROR: cubi_add(), ran out of room in result variable");
	cubi_free_d(&d);
	cubi_free_d(&ccpy);
}

/**
 * Divide cubi numbers, storing the result in the third (a / b = c)
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 * @param: (cubi*) c
 */
__device__
void cubi_div(cubi* a, cubi* b, cubi* c) {
	int sa = cubi_size(a);
	int sb = cubi_size(b);
	int sc = cubi_size(c);

	if (sa != sb || sa != sc) {
		printf("ERROR: cubi_div(a, b, c) requires that size of a,b,c are equal\n");
		return;
	}

	// Clean out c
	for (int i = 0; i < sc; i++)
		c->data[i] = 0;

	cubi zero, R, Rcpy, BxRES, RES, OVER;
	cubi_init_d(&zero, sb);
	if (cubi_cmp(&zero, b) == 0) {
		printf("ERROR: cubi_div(), divide by zero.\n");
		return;
	}

	cubi_init_d(&BxRES, sa);
	cubi_init_d(&R, sa);
	cubi_init_d(&Rcpy, sa);
	cubi_init_d(&RES, sa);
	cubi_init_d(&OVER, sa);
	cubi_copy(a, &R);
	int magB = cubi_magnitude(b);
	int carry = 0; int overshot = 0;
	while (cubi_cmp(&R, b) >= 0) {
		int magR = cubi_magnitude(&R);

		// Inital guess of c[magR - magB]
		long int res = R.data[magR] / b->data[magB];
		if (res == 0) {
			res = (R.data[magR] * 1000000) / b->data[magB];
			carry = 1;
		}
		if (res <= 0) { res *= -1; res++; }
		c->data[magR - magB - carry] += res - overshot;
		RES.data[magR - magB - carry] = res - overshot;
		cubi_mult(b, &RES, &BxRES);
		cubi_copy(&R, &Rcpy);
		if (cubi_cmp(&R, &BxRES) < 0) { // Then we overshot. Add remove from c
			c->data[magR - magB - carry] -= res - overshot;
//			overshot++;
			cubi_sub(&BxRES, &R, &OVER);
			int magOver = cubi_magnitude(&OVER);
			overshot += OVER.data[magOver] / b->data[magB] + 1; // add estimate for overshoot
		} else {
			cubi_sub(&Rcpy, &BxRES, &R);
			overshot = 0;
		}
		RES.data[magR - magB - carry] = 0;
		carry = 0;
	}

	cubi_free_d(&zero);
	cubi_free_d(&R);
	cubi_free_d(&Rcpy);
	cubi_free_d(&RES);
	cubi_free_d(&BxRES);
	cubi_free_d(&OVER);
}

/**
 * Calculate modulus of a cubi number: a (mod b) = c
 *
 * @param: (cubi*) a
 * @param: (cubi*) b
 * @param: (cubi*) c
 */
__device__
void cubi_mod(cubi* a, cubi* b, cubi* c) {
	int sa = cubi_size(a);
	int sb = cubi_size(b);
	int sc = cubi_size(c);

	if (sa != sb || sa != sc) {
		printf("ERROR: cubi_mod(a, b, c) requires that size of a,b,c are equal\n");
		return;
	}

	// Clean out c
	for (int i = 0; i < sc; i++)
		c->data[i] = 0;

	cubi Q, D;
	cubi_init_d(&D, sa);
	cubi_init_d(&Q, sa);
	cubi_div(a, b, &Q);
	cubi_mult(b, &Q, &D);
	cubi_sub(a, &D, c);

	cubi_free_d(&Q);
	cubi_free_d(&D);
}
