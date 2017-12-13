/* Code to test cubi.h class (Cuda Big Int)
 *
 * Andrew Corum, Dec 2017
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "cubi.h"

void testInit();
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
	testMagnitude();
	testCompare();
	testAdd();
	testSub();
	testMult();
	testDiv();
	testMod();

	fprintf(stderr, "All tests passed!\n\n");
	return 0;
}

/**
 * Tests for cubi modulus
 */
void testMod() {
	fprintf(stderr, "- Testing modulus...\n");

	cubi a, b, c, d;
	cubi_init(&a, 12);
	cubi_init(&b, 12);
	cubi_init(&c, 12);
	cubi_init(&d, 12);

	fprintf(stderr, "  + Small numbers... ");
	cubi_set_str(&a, (char*) "7154");
	cubi_set_str(&b, (char*) "98");
	cubi_set_str(&d, (char*) "0");
	cubi_mod(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Medium length numbers... ");
	cubi_set_str(&a, (char*) "56185479684165");
	cubi_set_str(&b, (char*) "862937951");
	cubi_set_str(&d, (char*) "452632506");
	cubi_mod(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Large numbers... ");
	cubi_set_str(&a, (char*) "159028561088562700000737018131892930468");
	cubi_set_str(&b, (char*) "63987234234780923458996723459");
	cubi_set_str(&d, (char*) "61879679652393658374941427468");
	cubi_mod(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);
	cubi_free(&c);
	cubi_free(&d);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Tests for cubi division
 */
void testDiv() {
	fprintf(stderr, "- Testing division...\n");

	cubi a, b, c, d;
	cubi_init(&a, 12);
	cubi_init(&b, 12);
	cubi_init(&c, 12);
	cubi_init(&d, 12);

	fprintf(stderr, "  + Evenly divide small numbers... ");
	cubi_set_str(&a, (char*) "738");
	cubi_set_str(&b, (char*) "41");
	cubi_set_str(&d, (char*) "18");
	cubi_div(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Evenly divide medium length numbers... ");
	cubi_set_str(&d, (char*) "501210");
	cubi_set_str(&b, (char*) "72342320");
	cubi_set_str(&a, (char*) "36258694207200");
	cubi_div(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Evenly divide large numbers... ");
	cubi_set_str(&b, (char*) "6780923458996723459");
	cubi_set_str(&d, (char*) "23452345694533453452");
	cubi_set_str(&a, (char*) "159028561088562700000737018131892930468");
	cubi_div(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Evenly divide large numbers reversed... ");
	cubi_set_str(&d, (char*) "6780923458996723459");
	cubi_set_str(&b, (char*) "23452345694533453452");
	cubi_set_str(&a, (char*) "159028561088562700000737018131892930468");
	cubi_div(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Divide small numbers with remainder... ");
	cubi_set_str(&a, (char*) "738");
	cubi_set_str(&b, (char*) "47");
	cubi_set_str(&d, (char*) "15");
	cubi_div(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Divide large numbers with remainder... ");
	cubi_set_str(&b, (char*) "7234593223");
	cubi_set_str(&d, (char*) "21981686625169734163053305109");
	cubi_set_str(&a, (char*) "159028561088562700000737018131892930468");
	cubi_div(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Divide large numbers with remainder... ");
	cubi_set_str(&b, (char*) "12723423");
	cubi_set_str(&d, (char*) "7096015819456526607021");
	cubi_set_str(&a, (char*) "90285610885637018131892930468");
	cubi_div(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Divide large numbers with remainder... ");
	cubi_set_str(&b, (char*) "12723423348923489792348923484");
	cubi_set_str(&d, (char*) "7");
	cubi_set_str(&a, (char*) "90285610885637018131892930468");
	cubi_div(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);
	cubi_free(&c);
	cubi_free(&d);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Tests for cubi multiplication
 */
void testMult() {
	fprintf(stderr, "- Testing multiplication...\n");

	cubi a, b, c, d;
	cubi_init(&a, 12);
	cubi_init(&b, 10);
	cubi_init(&c, 12);
	cubi_init(&d, 12);

	fprintf(stderr, "  + Small numbers... ");
	cubi_set_str(&a, (char*) "73");
	cubi_set_str(&b, (char*) "98");
	cubi_set_str(&d, (char*) "7154");
	cubi_mult(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Medium length numbers... ");
	cubi_set_str(&a, (char*) "501210");
	cubi_set_str(&b, (char*) "72342320");
	cubi_set_str(&d, (char*) "36258694207200");
	cubi_mult(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Large numbers... ");
	cubi_set_str(&a, (char*) "6780923458996723459");
	cubi_set_str(&b, (char*) "23452345694533453452");
	cubi_set_str(&d, (char*) "159028561088562700000737018131892930468");
	cubi_mult(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);
	cubi_free(&c);
	cubi_free(&d);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Tests for cubi subtraction
 */
void testSub() {
	fprintf(stderr, "- Testing subtraction...\n");

	cubi a, b, c, d;
	cubi_init(&a, 12);
	cubi_init(&b, 12);
	cubi_init(&c, 12);
	cubi_init(&d, 12);

	fprintf(stderr, "  + Subtract zero... ");
	cubi_set_str(&a, (char*) "7234");
	cubi_set_str(&b, (char*) "0");
	cubi_set_str(&d, (char*) "7234");
	cubi_sub(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Subtract small numbers... ");
	cubi_set_str(&a, (char*) "41");
	cubi_set_str(&b, (char*) "28");
	cubi_set_str(&d, (char*) "13");
	cubi_sub(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Subtract medium length numbers... ");
	cubi_set_str(&d, (char*) "500200");
	cubi_set_str(&b, (char*) "700900");
	cubi_set_str(&a, (char*) "1201100");
	cubi_sub(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Subtract long numbers... ");
	cubi_set_str(&d, (char*) "12389074574632345");
	cubi_set_str(&b, (char*) "3478934635734");
	cubi_set_str(&a, (char*) "12392553509268079");
	cubi_sub(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Even bigger subtraction problem... ");
	cubi_set_str(&d, (char*) "12389074574632345");
	cubi_set_str(&b, (char*) "159028561088562700000737018131892930468");
	cubi_set_str(&a, (char*) "159028561088562700000749407206467562813");
	cubi_sub(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);
	cubi_free(&c);
	cubi_free(&d);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Tests for cubi addition
 */
void testAdd() {
	fprintf(stderr, "- Testing addition...\n");

	cubi a, b, c, d;
	cubi_init(&a, 12);
	cubi_init(&b, 10);
	cubi_init(&c, 12);
	cubi_init(&d, 12);

	fprintf(stderr, "  + '7234 + 0 = 7234'... ");
	cubi_set_str(&a, (char*) "7234");
	cubi_set_str(&b, (char*) "0");
	cubi_set_str(&d, (char*) "7234");
	cubi_add(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + '13 + 28 = 41'... ");
	cubi_set_str(&a, (char*) "13");
	cubi_set_str(&b, (char*) "28");
	cubi_set_str(&d, (char*) "41");
	cubi_add(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + '500200 + 700900 = 1201100'... ");
	cubi_set_str(&a, (char*) "500200");
	cubi_set_str(&b, (char*) "700900");
	cubi_set_str(&d, (char*) "1201100");
	cubi_add(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + '12389074574632345 + 3478934635734 = 12392553509268079'... ");
	cubi_set_str(&a, (char*) "12389074574632345");
	cubi_set_str(&b, (char*) "3478934635734");
	cubi_set_str(&d, (char*) "12392553509268079");
	cubi_add(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Even bigger addition problem... ");
	cubi_set_str(&a, (char*) "12389074574632345");
	cubi_set_str(&b, (char*) "159028561088562700000737018131892930468");
	cubi_set_str(&d, (char*) "159028561088562700000749407206467562813");
	cubi_add(&a, &b, &c);
	assert(cubi_cmp(&c, &d) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);
	cubi_free(&c);
	cubi_free(&d);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Run tests for comparing numbers
 * */
void testCompare() {
	fprintf(stderr, "- Testing compare...\n");

	cubi a, b;
	cubi_init(&a, 12);
	cubi_init(&b, 3);

	fprintf(stderr, "  + 0 == 0... ");
	assert(cubi_cmp(&a, &b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + 0 < 123... ");
	cubi_set_str(&b, (char*) "123");
	assert(cubi_cmp(&a, &b) == -1);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + 48230434 > 123... ");
	cubi_set_str(&a, (char*) "48230434");
	assert(cubi_cmp(&a, &b) == 1);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + 45678123912390 == 45678123912390... ");
	cubi_set_str(&a, (char*) "45678123912390");
	cubi_set_str(&b, (char*) "45678123912390");
	assert(cubi_cmp(&a, &b) == 0);
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Run tests for magnitude
 */
void testMagnitude() {
	fprintf(stderr, "- Testing magnitude...\n");

	cubi a, b;
	cubi_init(&a, 12);
	cubi_init(&b, 3);

	fprintf(stderr, "  + Empty cubi... ");
	assert(cubi_magnitude(&a) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Small cubi (still mag 0)... ");
	cubi_set_str(&b, (char*) "2343");
	assert(cubi_magnitude(&b) == 0);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + med cubi (mag 1)... "); 
	cubi_set_str(&a, (char*) "12345612345");
	assert(cubi_magnitude(&a) == 1);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + large cubi (mag 3)... "); 
	cubi_set_str(&a, (char*) "123456123456123456123456");
	assert(cubi_magnitude(&a) == 3);
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);

	fprintf(stderr, "  Passed!\n\n");
}

/**
 * Run tests for Init, get size, set str, get str
 */
void testInit() {
	fprintf(stderr, "- Testing initializers...\n");

	cubi a, b;
	cubi_init(&a, 12);
	cubi_init(&b, 3);

	fprintf(stderr, "  + Size of a (12)... ");
	assert(cubi_size(&a) == 12);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + Size of b (3)... ");
	assert(cubi_size(&b) == 3);
	fprintf(stderr, "✔\n");

	fprintf(stderr, "  + cubi_set_str(), cubi_get_str()... "); 
	cubi_set_str(&a, (char*) "999999999888888887777777666666555554444333140714042285546");
	fprintf(stderr, "\n    a = 999999999888888887777777666666555554444333140714042285546 == %s ", cubi_get_str(&a));
	fprintf(stderr, "✔\n");

	cubi_set_str(&b, (char*) "140714042285546");
	fprintf(stderr, "    b = 140714042285546 == %s ", cubi_get_str(&b));
	fprintf(stderr, "✔\n");

	cubi_set_str(&a, (char*) "47");
	fprintf(stderr, "    a = 47 == %s ", cubi_get_str(&a));
	fprintf(stderr, "✔\n");

	cubi_free(&a);
	cubi_free(&b);

	fprintf(stderr, "  Passed!\n\n");
}
