"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input number unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers and return the result."""
    return x + y


def neg(x: float) -> float:
    """Negate a number and return the result."""
    return float(-x)


def lt(x: float, y: float) -> bool:
    """Check if the first operand is less than the second. Return True if so, False otherwise."""
    return x < y


def gt(x: float, y: float) -> bool:
    """Check if the first operand is greater than the second. Return True if so, False otherwise."""
    return x > y


def eq(x: float, y: float) -> bool:
    """Check if two numbers are equal. Return True if so, False otherwise."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close using absolute error.

    Args:
    ----
        x (float): First number to compare.
        y (float): Second number to compare.

    Returns:
    -------
        bool: True if the absolute difference between x and y is less than 1e-2, False otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    The sigmoid function is defined as:
    f(x) = 1 / exp(-x) if x >= 0
         = exp(x) / (1 + exp(x)) if x < 0

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The sigmoid of x.

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU function.

    The ReLU function is defined as:
    f(x) = x if x >= 0 else 0

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The ReLU of x.

    """
    return float(x) if x >= 0 else 0.0


def log(x: float) -> float:
    """Compute the logarithm of the input number.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential of the input number."""
    # TODO: might cause precision issues
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the inverse of the input number."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Compute the derivative of the log function with respect to the first argument times the second."""
    return d * inv(x)


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of the inverse function with respect to the first argument times the second."""
    return -d * (1.0 / x) ** 2


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of the ReLU function with respect to the first argument times the second."""
    return d * (1 if x > 0 else 0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float], xs: Iterable[float]) -> list[float]:
    """Apply a function to each element of an Iterable. Return a new list with the results."""
    xs = iter(xs)
    return [f(x) for x in xs]


def zipWith(
    f: Callable[[float, float], float],
    xs: Iterable[float],
    ys: Iterable[float],
) -> list[float]:
    """Apply a function to corresponding elements of two lists. Return a new list with the results."""
    xs = iter(xs)
    ys = iter(ys)
    return [f(x, y) for x, y in zip(xs, ys)]


def reduce(f: Callable[[T, T], T], start: T) -> Callable[[Iterable[T]], T]:
    """Create a reducer function that applies a binary function to an iterable.

    Args:
    ----
        f (Callable[[T, T], T]): A binary function to apply to the elements.
        start (Optional[T]): An optional initial value. If None, the first element of the iterable is used.

    Returns:
    -------
        Callable[[Iterable[T]], T]: A function that takes an iterable and returns the reduced result.

    Raises:
    ------
        ValueError: If the iterable is empty and no start value is provided.

    Example:
    -------
        >>> sum_reducer = reduce(lambda x, y: x + y)
        >>> sum_reducer([1, 2, 3, 4])
        10

    """

    def _reduce(ls: Iterable[T]) -> T:
        val = start
        for x in ls:
            val = f(val, x)
        return val

    return _reduce


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list."""
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists."""
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list."""
    _reduce = reduce(add, 0)
    return _reduce(xs)


def prod(xs: Iterable[float]) -> float:
    """Multiply a list."""
    _reduce = reduce(mul, 1)
    return _reduce(xs)
