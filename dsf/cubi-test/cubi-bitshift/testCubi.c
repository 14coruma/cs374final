/* Code to test cubi.h class (Cuda Big Int)
 *
 * Andrew Corum, Dec 2017
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "cubi-c.h"

void testInit();
void testShiftRight();
void testShiftLeft();
void testCompare();
void testMagnitude();
void testAdd();
void testSub();
void testMult();
void testDiv();
void testMod();
/**
 * Run cubi.h data type tests
 */
int main(int argc, char** argv) {
	fprintf(stderr, "Testing cubi.h:\n");

	testInit();
//	testShiftRight();
	testShiftLeft();
//	testMagnitude();
	testCompare();
	testAdd();
	testSub();
	testMult();
	testDiv();
//	testMod();

	fprintf(stderr, "All tests passed!\n\n");
	return 0;
}

/**
 * Tests for cubi modulus
 */
/*void testMod() {
	fprintf(stderr, "- Testing modulus...\n");

	cubi a, b, c, d;
	cubi_init(&a, 12);
	cubi_init(&b, 12);
	cubi_init(&c, 12);
	cubi_init(&d, 12);

	fprintf(stderr, "  + Small numbers... ");
	cubi_set_str_bin(&a, (char*) "7154");
	cubi_set_str_bin(&b, (char*) "98");
	cubi_set_str_bin(&d, (char*) "0");
	cubi_mod(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Medium length numbers... ");
	cubi_set_str_bin(&a, (char*) "56185479684165");
	cubi_set_str_bin(&b, (char*) "862937951");
	cubi_set_str_bin(&d, (char*) "452632506");
	cubi_mod(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Large numbers... ");
	cubi_set_str_bin(&a, (char*) "159028561088562700000737018131892930468");
	cubi_set_str_bin(&b, (char*) "63987234234780923458996723459");
	cubi_set_str_bin(&d, (char*) "61879679652393658374941427468");
	cubi_mod(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);
	cubi_free(&c);
	cubi_free(&d);

	fprintf(stderr, "  Passed!\n\n");
}*/

/**
 * Tests for cubi division
 */
void testDiv() {
	fprintf(stderr, "- Testing division...\n");

	cubi a, b, c, d, r;
	a = (cubi) malloc(SIZE * sizeof(int));
	b = (cubi) malloc(SIZE * sizeof(int));
	c = (cubi) malloc(SIZE * sizeof(int));
	d = (cubi) malloc(SIZE * sizeof(int));
	r = (cubi) malloc(SIZE * sizeof(int));
	cubi_init(a);
	cubi_init(b);
	cubi_init(c);
	cubi_init(d);
	cubi_init(r);

	fprintf(stderr, "  + Small numbers evenly... ");
	cubi_set_str_bin(d, (char*) "10011");
	cubi_set_str_bin(b, (char*) "111");
	cubi_set_str_bin(a, (char*) "10000101");
	cubi_div(a, b, c, r);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Medium length numbers evenly... ");
	cubi_set_str_bin(d, (char*) "1111010010111011010");
	cubi_set_str_bin(b, (char*) "100010011111101101100110000");
	cubi_set_str_bin(a, (char*) "1000001111101000100010100101101001011011100000");
	cubi_div(a, b, c, r);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Large numbers evenly... ");
	cubi_set_str_bin(b, (char*) "101111000011010101011011111010110000110100111001101111100000011");
	cubi_set_str_bin(d, (char*) "10100010101110111011110000010011101000000010101000000011010001100");
	cubi_set_str_bin(a, (char*) "1110111101000111100100101101010100000000100011111100000001100100001000010110110101010000001110000111000000000000000011110100100");
	cubi_div(a, b, c, r);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");


	cubi_free(a);
	cubi_free(b);
	cubi_free(c);
	cubi_free(d);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Tests for cubi multiplication
 */
void testMult() {
	fprintf(stderr, "- Testing multiplication...\n");

	cubi a, b, c, d;
	a = (cubi) malloc(SIZE * sizeof(int));
	b = (cubi) malloc(SIZE * sizeof(int));
	c = (cubi) malloc(SIZE * sizeof(int));
	d = (cubi) malloc(SIZE * sizeof(int));
	cubi_init(a);
	cubi_init(b);
	cubi_init(c);
	cubi_init(d);

	fprintf(stderr, "  + Small numbers... ");
	cubi_set_str_bin(a, (char*) "10011");
	cubi_set_str_bin(b, (char*) "111");
	cubi_set_str_bin(d, (char*) "10000101");
	cubi_mult(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Medium length numbers... ");
	cubi_set_str_bin(a, (char*) "1111010010111011010");
	cubi_set_str_bin(b, (char*) "100010011111101101100110000");
	cubi_set_str_bin(d, (char*) "1000001111101000100010100101101001011011100000");
	cubi_mult(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Large numbers... ");
	cubi_set_str_bin(a, (char*) "101111000011010101011011111010110000110100111001101111100000011");
	cubi_set_str_bin(b, (char*) "10100010101110111011110000010011101000000010101000000011010001100");
	cubi_set_str_bin(d, (char*) "1110111101000111100100101101010100000000100011111100000001100100001000010110110101010000001110000111000000000000000011110100100");
	cubi_mult(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(a);
	cubi_free(b);
	cubi_free(c);
	cubi_free(d);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Tests for cubi subtraction
 */
void testSub() {
	fprintf(stderr, "- Testing subtraction...\n");

	cubi a, b, c, d;
	a = (cubi) malloc(SIZE * sizeof(int));
	b = (cubi) malloc(SIZE * sizeof(int));
	c = (cubi) malloc(SIZE * sizeof(int));
	d = (cubi) malloc(SIZE * sizeof(int));
	cubi_init(a);
	cubi_init(b);
	cubi_init(c);
	cubi_init(d);

	fprintf(stderr, "  + '7234 - 0 = 7234'... ");
	cubi_set_str_bin(a, (char*) "100010011111110111001100010");
	cubi_set_str_bin(b, (char*) "0");
	cubi_set_str_bin(d, (char*) "100010011111110111001100010");
	cubi_sub(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + '41 - 13 = 28'... ");
	cubi_set_str_bin(d, (char*) "1101");
	cubi_set_str_bin(b, (char*) "11100");
	cubi_set_str_bin(a, (char*) "101001");
	cubi_sub(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + '556002000 = 74600900 - 630602900'... ");
	cubi_set_str_bin(b, (char*) "100001001000111110101011010000");
	cubi_set_str_bin(d, (char*) "100011100100101000111000100");
	cubi_set_str_bin(a, (char*) "100101100101100011110010010100");
	cubi_sub(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Big num - really big num... ");
	cubi_set_str_bin(d, (char*) "10100101010101110110101011010101010111010101010101010101010111011110000010011101000000010101000000011010001100");
	cubi_set_str_bin(b, (char*) "1110111101000111100100101101010100000000100011111100000001100100001000010110110101010000001110000111000000000000000011110100100");
	cubi_set_str_bin(a, (char*) "1110111101000111111001011000000010110101111110100110111100001110110011000001110001000000100001101111000010101000000111000110000");
	cubi_sub(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(a);
	cubi_free(b);
	cubi_free(c);
	cubi_free(d);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Tests for cubi addition
 */
void testAdd() {
	fprintf(stderr, "- Testing addition...\n");

	cubi a, b, c, d;
	a = (cubi) malloc(SIZE * sizeof(int));
	b = (cubi) malloc(SIZE * sizeof(int));
	c = (cubi) malloc(SIZE * sizeof(int));
	d = (cubi) malloc(SIZE * sizeof(int));
	cubi_init(a);
	cubi_init(b);
	cubi_init(c);
	cubi_init(d);

	fprintf(stderr, "  + '7234 + 0 = 7234'... ");
	cubi_set_str_bin(a, (char*) "100010011111110111001100010");
	cubi_set_str_bin(b, (char*) "0");
	cubi_set_str_bin(d, (char*) "100010011111110111001100010");
	cubi_add(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + '13 + 28 = 41'... ");
	cubi_set_str_bin(a, (char*) "1101");
	cubi_set_str_bin(b, (char*) "11100");
	cubi_set_str_bin(d, (char*) "101001");
	cubi_add(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + '556002000 + 74600900 = 630602900'... ");
	cubi_set_str_bin(a, (char*) "100001001000111110101011010000");
	cubi_set_str_bin(b, (char*) "100011100100101000111000100");
	cubi_set_str_bin(d, (char*) "100101100101100011110010010100");
	cubi_add(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Big num + really big num... ");
	cubi_set_str_bin(a, (char*) "10100101010101110110101011010101010111010101010101010101010111011110000010011101000000010101000000011010001100");
	cubi_set_str_bin(b, (char*) "1110111101000111100100101101010100000000100011111100000001100100001000010110110101010000001110000111000000000000000011110100100");
	cubi_set_str_bin(d, (char*) "1110111101000111111001011000000010110101111110100110111100001110110011000001110001000000100001101111000010101000000111000110000");
	cubi_add(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

/*	fprintf(stderr, "  + '12389074574632345 + 3478934635734 = 12392553509268079'... ");
	cubi_set_str_bin(a, (char*) "12389074574632345");
	cubi_set_str_bin(b, (char*) "3478934635734");
	cubi_set_str_bin(d, (char*) "12392553509268079");
	cubi_add(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Even bigger addition problem... ");
	cubi_set_str_bin(a, (char*) "12389074574632345");
	cubi_set_str_bin(b, (char*) "159028561088562700000737018131892930468");
	cubi_set_str_bin(d, (char*) "159028561088562700000749407206467562813");
	cubi_add(a, b, c);
	assert(cubi_cmp(c, d) == 0);
	fprintf(stderr, "✔\n");*/

	cubi_free(a);
	cubi_free(b);
	cubi_free(c);
	cubi_free(d);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Run tests for a right left
 */
void testShiftLeft() {
	fprintf(stderr, "- Testing shift left...\n");
	cubi a, b, zero;
	a = (cubi) malloc(SIZE * sizeof(int));
	b = (cubi) malloc(SIZE * sizeof(int));
	cubi_init(a);
	cubi_init(b);

	fprintf(stderr, "  + Shift empty cubi left... ");
	cubi_shift_left(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Shift small even cubi left... ");
	cubi_set_str_bin(b, (char*) "11101010");
	cubi_set_str_bin(a, (char*) "1110101");
	cubi_shift_left(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Shift medium length even cubi left... ");
	cubi_set_str_bin(b, (char*) "110101001110101010011010110000");
	cubi_set_str_bin(a, (char*) "11010100111010101001101011000");
	cubi_shift_left(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Shift large length even cubi left... ");
	cubi_set_str_bin(b, (char*) "1101010101011010101010101100101010110101010101010101010011010");
	cubi_set_str_bin(a, (char*) "110101010101101010101010110010101011010101010101010101001101");
	cubi_shift_left(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

/*	fprintf(stderr, "  + Shift even Larger even cubi left... ");
	cubi_set_str_bin(b, (char*) "23872435098723450982345084352");
	cubi_set_str_bin(a, (char*) "11936217549361725491172542176");
	cubi_shift_left(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Shift even Larger even cubi left... ");
	cubi_set_str_bin(b, (char*) "437684350872345089723459083245987234598723458978");
	cubi_set_str_bin(a, (char*) "218842175436172544861729541622993617299361729489");
	cubi_shift_left(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");*/

	cubi_free(a);
	cubi_free(b);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Run tests for a right shift
 */
void testShiftRight() {
	fprintf(stderr, "- Testing right...\n");
	cubi a, b, zero;
	a = (cubi) malloc(SIZE * sizeof(int));
	b = (cubi) malloc(SIZE * sizeof(int));
	cubi_init(a);
	cubi_init(b);

	fprintf(stderr, "  + Shift empty cubi right... ");
	cubi_shift_right(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Shift small even cubi right... ");
	cubi_set_str_bin(a, (char*) "46");
	cubi_set_str_bin(b, (char*) "23");
	cubi_shift_right(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Shift medium length even cubi right... ");
	cubi_set_str_bin(a, (char*) "4664364634");
	cubi_set_str_bin(b, (char*) "2332182317");
	cubi_shift_right(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Shift large length even cubi right... ");
	cubi_set_str_bin(a, (char*) "34987234978234");
	cubi_set_str_bin(b, (char*) "17493617489117");
	cubi_shift_right(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Shift even Larger even cubi right... ");
	cubi_set_str_bin(a, (char*) "23872435098723450982345084352");
	cubi_set_str_bin(b, (char*) "11936217549361725491172542176");
	cubi_shift_right(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Shift even Larger even cubi right... ");
	cubi_set_str_bin(a, (char*) "437684350872345089723459083245987234598723458978");
	cubi_set_str_bin(b, (char*) "218842175436172544861729541622993617299361729489");
	cubi_shift_right(a);
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(a);
	cubi_free(b);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Run tests for comparing numbers
 * */
void testCompare() {
	fprintf(stderr, "- Testing compare...\n");

	cubi a, b;
	a = (cubi) malloc(SIZE * sizeof(int));
	b = (cubi) malloc(SIZE * sizeof(int));
	cubi_init(a);
	cubi_init(b);

	fprintf(stderr, "  + 0 == 0... ");
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + 0 < 123... ");
	cubi_set_str_bin(b, (char*) "1111011");
	assert(cubi_cmp(a, b) == -1);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + big num > 123... ");
	cubi_set_str_bin(a, (char*) "10110111111111000010");
	assert(cubi_cmp(a, b) == 1);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + same big num == same big num... ");
	cubi_set_str_bin(a, (char*) "10110111111111000010");
	cubi_set_str_bin(b, (char*) "10110111111111000010");
	assert(cubi_cmp(a, b) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(a);
	cubi_free(b);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Run tests for magnitude
 */
/*void testMagnitude() {
	fprintf(stderr, "- Testing magnitude...\n");

	cubi a, b;
	cubi_init(&a, 12);
	cubi_init(&b, 3);

	fprintf(stderr, "  + Empty cubi... ");
	assert(cubi_magnitude(&a) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Small cubi (still mag 0)... ");
	cubi_set_str_bin(&b, (char*) "2343");
	assert(cubi_magnitude(&b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + med cubi (mag 1)... "); 
	cubi_set_str_bin(&a, (char*) "12345612345");
	assert(cubi_magnitude(&a) == 1);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + large cubi (mag 3)... "); 
	cubi_set_str_bin(&a, (char*) "123456123456123456123456");
	assert(cubi_magnitude(&a) == 3);
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);

	fprintf(stderr, "  Passed!\n\n");
}*/

/**
 * Run tests for Init, get size, set str, get str
 */
void testInit() {
	fprintf(stderr, "- Testing initializers...\n");

	cubi a, b;
	a = (cubi) malloc(SIZE * sizeof(int));
	b = (cubi) malloc(SIZE * sizeof(int));
	cubi_init(a);
	cubi_init(b);

	fprintf(stderr, "  + Size of a... ");
	assert(cubi_size(a) == 12);
	fprintf(stderr, "✔\n");

	cubi_set_str_bin(a, (char*) "101");
	fprintf(stderr, "    a = 101 = 5 == ");
	fprintf(stderr, "✔\n");

	cubi_set_str_bin(a, (char*) "110110101011010101001101");
	fprintf(stderr, "    a = 14333261... ");
	fprintf(stderr, "✔\n");

	cubi_set_str_bin(a, (char*) "11010101011010100101010110101010011");
	fprintf(stderr, "    a = 28644126035... ");
	fprintf(stderr, "✔\n");

	cubi_set_str_bin(b, (char*) "1101010101011010101010101100101010110101010101010101010011010");
	fprintf(stderr, "    a = 1921723508198124186... ");
	fprintf(stderr, "✔\n");

	fprintf(stderr, "    a = 159028561088562700000737018131892930468... ");
	cubi_set_str_bin(a, (char*) "1110111101000111100100101101010100000000100011111100000001100100001000010110110101010000001110000111000000000000000011110100100");
	fprintf(stderr, "✔\n");
/*	fprintf(stderr, "  + cubi_set_str_bin(), cubi_get_str()... "); 
	cubi_set_str_bin(a, (char*) "999999999888888887777777666666555554444333140714042285546");
	fprintf(stderr, "\n    a = 999999999888888887777777666666555554444333140714042285546 == ");
	fprintf(stderr, "✔\n");

	cubi_set_str_bin(b, (char*) "140714042285546");
	fprintf(stderr, "    b = 140714042285546 == %s ", cubi_get_str(b));
	fprintf(stderr, "✔\n");*/

	cubi_free(a);
	cubi_free(b);

	fprintf(stderr, "  Passed!\n\n");
}
