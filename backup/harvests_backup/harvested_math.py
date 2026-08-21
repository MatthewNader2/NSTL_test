# Auto-harvested from math
import typing

def math_acos(x: 'any') -> 'any_computed':
    """
    Return the arc cosine (measured in radians) of x.
    The result is between 0 and pi.
    Keywords: math, acos
    """
    import math
    output_var = math.acos(input_var)

def math_acosh(x: 'any') -> 'any_computed':
    """
    Return the inverse hyperbolic cosine of x.
    Keywords: math, acosh
    """
    import math
    output_var = math.acosh(input_var)

def math_asin(x: 'any') -> 'any_computed':
    """
    Return the arc sine (measured in radians) of x.
    The result is between -pi/2 and pi/2.
    Keywords: math, asin
    """
    import math
    output_var = math.asin(input_var)

def math_asinh(x: 'any') -> 'any_computed':
    """
    Return the inverse hyperbolic sine of x.
    Keywords: math, asinh
    """
    import math
    output_var = math.asinh(input_var)

def math_atan(x: 'any') -> 'any_computed':
    """
    Return the arc tangent (measured in radians) of x.
    The result is between -pi/2 and pi/2.
    Keywords: math, atan
    """
    import math
    output_var = math.atan(input_var)

def math_atan2(y: 'any') -> 'any_computed':
    """
    Return the arc tangent (measured in radians) of y/x.
    Unlike atan(y/x), the signs of both x and y are considered.
    Keywords: math, atan2
    """
    import math
    output_var = math.atan2(input_var)

def math_atanh(x: 'any') -> 'any_computed':
    """
    Return the inverse hyperbolic tangent of x.
    Keywords: math, atanh
    """
    import math
    output_var = math.atanh(input_var)

def math_cbrt(x: 'any') -> 'any_computed':
    """
    Return the cube root of x.
    Keywords: math, cbrt
    """
    import math
    output_var = math.cbrt(input_var)

def math_ceil(x: 'any') -> 'any_computed':
    """
    Return the ceiling of x as an Integral.
    This is the smallest integer >= x.
    Keywords: math, ceil
    """
    import math
    output_var = math.ceil(input_var)

def math_comb(n: 'any') -> 'any_computed':
    """
    Number of ways to choose k items from n items without repetition and without order.
    Evaluates to n! / (k! * (n - k)!) when k <= n and evaluates
    to zero when k > n.
    Keywords: math, comb
    """
    import math
    output_var = math.comb(input_var)

def math_copysign(x: 'any') -> 'any_computed':
    """
    Return a float with the magnitude (absolute value) of x but the sign of y.
    On platforms that support signed zeros, copysign(1.0, -0.0)
    returns -1.0.
    Keywords: math, copysign
    """
    import math
    output_var = math.copysign(input_var)

def math_cos(x: 'any') -> 'any_computed':
    """
    Return the cosine of x (measured in radians).
    Keywords: math, cos
    """
    import math
    output_var = math.cos(input_var)

def math_cosh(x: 'any') -> 'any_computed':
    """
    Return the hyperbolic cosine of x.
    Keywords: math, cosh
    """
    import math
    output_var = math.cosh(input_var)

def math_degrees(x: 'any') -> 'any_computed':
    """
    Convert angle x from radians to degrees.
    Keywords: math, degrees
    """
    import math
    output_var = math.degrees(input_var)

def math_dist(p: 'any') -> 'any_computed':
    """
    Return the Euclidean distance between two points p and q.
    The points should be specified as sequences (or iterables) of
    coordinates.  Both inputs must have the same dimension.
    Keywords: math, dist
    """
    import math
    output_var = math.dist(input_var)

def math_erf(x: 'any') -> 'any_computed':
    """
    Error function at x.
    Keywords: math, erf
    """
    import math
    output_var = math.erf(input_var)

def math_erfc(x: 'any') -> 'any_computed':
    """
    Complementary error function at x.
    Keywords: math, erfc
    """
    import math
    output_var = math.erfc(input_var)

def math_exp(x: 'any') -> 'any_computed':
    """
    Return e raised to the power of x.
    Keywords: math, exp
    """
    import math
    output_var = math.exp(input_var)

def math_exp2(x: 'any') -> 'any_computed':
    """
    Return 2 raised to the power of x.
    Keywords: math, exp2
    """
    import math
    output_var = math.exp2(input_var)

def math_expm1(x: 'any') -> 'any_computed':
    """
    Return exp(x)-1.
    This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x.
    Keywords: math, expm1
    """
    import math
    output_var = math.expm1(input_var)

def math_fabs(x: 'any') -> 'any_computed':
    """
    Return the absolute value of the float x.
    Keywords: math, fabs
    """
    import math
    output_var = math.fabs(input_var)

def math_factorial(n: 'any') -> 'any_computed':
    """
    Find n!.
    Raise a ValueError if x is negative or non-integral.
    Keywords: math, factorial
    """
    import math
    output_var = math.factorial(input_var)

def math_floor(x: 'any') -> 'any_computed':
    """
    Return the floor of x as an Integral.
    This is the largest integer <= x.
    Keywords: math, floor
    """
    import math
    output_var = math.floor(input_var)

def math_fma(x: 'any') -> 'any_computed':
    """
    Fused multiply-add operation.
    Compute (x * y) + z with a single round.
    Keywords: math, fma
    """
    import math
    output_var = math.fma(input_var)

def math_fmod(x: 'any') -> 'any_computed':
    """
    Return fmod(x, y), according to platform C.
    x % y may differ.
    Keywords: math, fmod
    """
    import math
    output_var = math.fmod(input_var)

def math_frexp(x: 'any') -> 'any_computed':
    """
    Return the mantissa and exponent of x, as pair (m, e).
    m is a float and e is an int, such that x = m * 2.**e.
    If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0.
    Keywords: math, frexp
    """
    import math
    output_var = math.frexp(input_var)

def math_fsum(seq: 'any') -> 'any_computed':
    """
    Return an accurate floating-point sum of values in the iterable seq.
    Assumes IEEE-754 floating-point arithmetic.
    Keywords: math, fsum
    """
    import math
    output_var = math.fsum(input_var)

def math_gamma(x: 'any') -> 'any_computed':
    """
    Gamma function at x.
    Keywords: math, gamma
    """
    import math
    output_var = math.gamma(input_var)

def math_gcd(integers: 'any') -> 'any_computed':
    """
    Greatest Common Divisor.
    Keywords: math, gcd
    """
    import math
    output_var = math.gcd(input_var)

def math_isclose(a: 'any') -> 'any_computed':
    """
    Determine whether two floating-point numbers are close in value.
      rel_tol
        maximum difference for being considered "close", relative to the
        magnitude of the input values
    Keywords: math, isclose
    """
    import math
    output_var = math.isclose(input_var)

def math_isfinite(x: 'any') -> 'any_computed':
    """
    Return True if x is neither an infinity nor a NaN, and False otherwise.
    Keywords: math, isfinite
    """
    import math
    output_var = math.isfinite(input_var)

def math_isinf(x: 'any') -> 'any_computed':
    """
    Return True if x is a positive or negative infinity, and False otherwise.
    Keywords: math, isinf
    """
    import math
    output_var = math.isinf(input_var)

def math_isnan(x: 'any') -> 'any_computed':
    """
    Return True if x is a NaN (not a number), and False otherwise.
    Keywords: math, isnan
    """
    import math
    output_var = math.isnan(input_var)

def math_isqrt(n: 'any') -> 'any_computed':
    """
    Return the integer part of the square root of the input.
    Keywords: math, isqrt
    """
    import math
    output_var = math.isqrt(input_var)

def math_lcm(integers: 'any') -> 'any_computed':
    """
    Least Common Multiple.
    Keywords: math, lcm
    """
    import math
    output_var = math.lcm(input_var)

def math_ldexp(x: 'any') -> 'any_computed':
    """
    Return x * (2**i).
    This is essentially the inverse of frexp().
    Keywords: math, ldexp
    """
    import math
    output_var = math.ldexp(input_var)

def math_lgamma(x: 'any') -> 'any_computed':
    """
    Natural logarithm of absolute value of Gamma function at x.
    Keywords: math, lgamma
    """
    import math
    output_var = math.lgamma(input_var)

def math_log10(x: 'any') -> 'any_computed':
    """
    Return the base 10 logarithm of x.
    Keywords: math, log10
    """
    import math
    output_var = math.log10(input_var)

def math_log1p(x: 'any') -> 'any_computed':
    """
    Return the natural logarithm of 1+x (base e).
    The result is computed in a way which is accurate for x near zero.
    Keywords: math, log1p
    """
    import math
    output_var = math.log1p(input_var)

def math_log2(x: 'any') -> 'any_computed':
    """
    Return the base 2 logarithm of x.
    Keywords: math, log2
    """
    import math
    output_var = math.log2(input_var)

def math_modf(x: 'any') -> 'any_computed':
    """
    Return the fractional and integer parts of x.
    Both results carry the sign of x and are floats.
    Keywords: math, modf
    """
    import math
    output_var = math.modf(input_var)

def math_nextafter(x: 'any') -> 'any_computed':
    """
    Return the floating-point value the given number of steps after x towards y.
    If steps is not specified or is None, it defaults to 1.
    Raises a TypeError, if x or y is not a double, or if steps is not an integer.
    Keywords: math, nextafter
    """
    import math
    output_var = math.nextafter(input_var)

def math_perm(n: 'any') -> 'any_computed':
    """
    Number of ways to choose k items from n items without repetition and with order.
    Evaluates to n! / (n - k)! when k <= n and evaluates
    to zero when k > n.
    Keywords: math, perm
    """
    import math
    output_var = math.perm(input_var)

def math_pow(x: 'any') -> 'any_computed':
    """
    Return x**y (x to the power of y).
    Keywords: math, pow
    """
    import math
    output_var = math.pow(input_var)

def math_prod(iterable: 'any') -> 'any_computed':
    """
    Calculate the product of all the elements in the input iterable.
    The default start value for the product is 1.
    When the iterable is empty, return the start value.  This function is
    Keywords: math, prod
    """
    import math
    output_var = math.prod(input_var)

def math_radians(x: 'any') -> 'any_computed':
    """
    Convert angle x from degrees to radians.
    Keywords: math, radians
    """
    import math
    output_var = math.radians(input_var)

def math_remainder(x: 'any') -> 'any_computed':
    """
    Difference between x and the closest integer multiple of y.
    Return x - n*y where n*y is the closest integer multiple of y.
    In the case where x is exactly halfway between two multiples of
    y, the nearest even value of n is used. The result is always exact.
    Keywords: math, remainder
    """
    import math
    output_var = math.remainder(input_var)

def math_sin(x: 'any') -> 'any_computed':
    """
    Return the sine of x (measured in radians).
    Keywords: math, sin
    """
    import math
    output_var = math.sin(input_var)

def math_sinh(x: 'any') -> 'any_computed':
    """
    Return the hyperbolic sine of x.
    Keywords: math, sinh
    """
    import math
    output_var = math.sinh(input_var)

def math_sqrt(x: 'any') -> 'any_computed':
    """
    Return the square root of x.
    Keywords: math, sqrt
    """
    import math
    output_var = math.sqrt(input_var)

def math_sumprod(p: 'any') -> 'any_computed':
    """
    Return the sum of products of values from two iterables p and q.
    Roughly equivalent to:
        sum(itertools.starmap(operator.mul, zip(p, q, strict=True)))
    Keywords: math, sumprod
    """
    import math
    output_var = math.sumprod(input_var)

def math_tan(x: 'any') -> 'any_computed':
    """
    Return the tangent of x (measured in radians).
    Keywords: math, tan
    """
    import math
    output_var = math.tan(input_var)

def math_tanh(x: 'any') -> 'any_computed':
    """
    Return the hyperbolic tangent of x.
    Keywords: math, tanh
    """
    import math
    output_var = math.tanh(input_var)

def math_trunc(x: 'any') -> 'any_computed':
    """
    Truncates the Real x to the nearest Integral toward 0.
    Uses the __trunc__ magic method.
    Keywords: math, trunc
    """
    import math
    output_var = math.trunc(input_var)

def math_ulp(x: 'any') -> 'any_computed':
    """
    Return the value of the least significant bit of the float x.
    Keywords: math, ulp
    """
    import math
    output_var = math.ulp(input_var)
