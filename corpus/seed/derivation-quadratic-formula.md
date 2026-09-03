---
title: Derivation of the quadratic formula
license: CC0-1.0
version: 2026-01-15
collection: seed-mathematics
doc_type: derivation
---

# Derivation of the quadratic formula

Claim: for a x^2 + b x + c = 0 with a not equal to 0,

    x = ( -b ± sqrt(b^2 - 4 a c) ) / (2 a)

## Steps

1. Divide by a, which is legal because a is not zero:

       x^2 + (b/a) x + c/a = 0

2. Move the constant to the right side:

       x^2 + (b/a) x = -c/a

3. Complete the square by adding the square of half the linear coefficient,
   (b / 2a)^2, to both sides:

       x^2 + (b/a) x + (b / 2a)^2 = (b / 2a)^2 - c/a

4. The left side is now a perfect square:

       (x + b / 2a)^2 = (b^2 - 4 a c) / (4 a^2)

5. Take the square root of both sides, keeping both signs:

       x + b / 2a = ± sqrt(b^2 - 4 a c) / (2 a)

6. Isolate x:

       x = ( -b ± sqrt(b^2 - 4 a c) ) / (2 a)

## The discriminant

The quantity b^2 - 4 a c decides the character of the roots: positive gives two
distinct real roots, zero gives one repeated real root, negative gives a
conjugate pair of complex roots. Step 5 is where that case split enters the
derivation; nothing before it depends on the sign.
