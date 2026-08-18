"""Conservative reusable solvers for exactly verifiable multiple-choice mathematics."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
import math
import re

from .options import parse_options


@dataclass(frozen=True)
class VerifiedMathAnswer:
    label: str
    value: Fraction
    solver: str
    proof: str


def _option_fraction(description: str) -> Fraction | None:
    text = description.strip().replace(",", "").replace("−", "-")
    match = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if match:
        return Fraction(int(match.group(1)), int(match.group(2)))
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return Fraction(text)
    number_words = {"none": 0, "one": 1, "two": 2, "three": 3, "four": 4}
    if text.lower() in number_words:
        return Fraction(number_words[text.lower()])
    return None


def _map_value(question: str, value: int | float | Fraction, solver: str, proof: str) -> VerifiedMathAnswer:
    expected = value if isinstance(value, Fraction) else Fraction(str(value))
    matches = [
        option.label
        for option in parse_options(question)
        if _option_fraction(option.description) == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"Computed {expected} does not map to exactly one offered option")
    return VerifiedMathAnswer(matches[0], expected, solver, proof)


def solve_verified_math(question: str) -> VerifiedMathAnswer | None:
    """Solve only recognized families with an exact programmatic proof."""
    text = " ".join(question.replace("−", "-").split())

    match = re.search(r"greatest common factor of (\d+) and (\d+)", text, re.I)
    if match:
        a, b = map(int, match.groups())
        value = math.gcd(a, b)
        return _map_value(question, value, "gcd", f"gcd({a}, {b}) = {value}")

    match = re.search(r"smallest positive integer with factors of (\d+), (\d+), and (\d+)", text, re.I)
    if match:
        values = list(map(int, match.groups()))
        result = math.lcm(*values)
        return _map_value(question, result, "lcm", f"lcm{tuple(values)} = {result}")

    match = re.search(r"ones digit of \$?([\d\s\\cdot·*]+)", text, re.I)
    if match:
        values = list(map(int, re.findall(r"\d+", match.group(1))))
        product = math.prod(values)
        return _map_value(question, product % 10, "ones_digit", f"product mod 10 = {product % 10}")

    match = re.search(r"remainder when \$?(\d+)\^\{?(\d+)\}?\s*\+(\d+)\$? is divided by \$?(\d+)", text, re.I)
    if match:
        base, exponent, addend, modulus = map(int, match.groups())
        result = (pow(base, exponent, modulus) + addend) % modulus
        return _map_value(question, result, "modular_power", f"({base}^{exponent}+{addend}) mod {modulus} = {result}")

    match = re.search(r"value of \$(\d+)\^2 \+ (\d+)\^2 \+ \\cdots \+ (\d+)\^2\$", text)
    if match:
        start, second, end = map(int, match.groups())
        if second != start + 1:
            return None
        result = sum(value * value for value in range(start, end + 1))
        return _map_value(question, result, "sum_squares", f"sum(k^2, k={start}..{end}) = {result}")

    match = re.search(r"How many numbers are in the list \$(\d+), (\d+), .*?, (\d+), (\d+) \?\$", text, re.I)
    if match:
        first, second, penultimate, last = map(int, match.groups())
        step = second - first
        if step <= 0 or last - penultimate != step or (last - first) % step:
            return None
        count = (last - first) // step + 1
        return _map_value(question, count, "arithmetic_sequence_count", f"({last}-{first})/{step}+1 = {count}")

    match = re.search(r"committees of (\d+) people can be chosen from a group of (\d+)", text, re.I)
    if match:
        selected, total = map(int, match.groups())
        result = math.comb(total, selected)
        return _map_value(question, result, "combinations", f"C({total},{selected}) = {result}")

    match = re.search(r"How many odd perfect squares are between (\d+) and (\d+)", text, re.I)
    if match:
        low, high = map(int, match.groups())
        roots = [n for n in range(math.isqrt(high) + 1) if low < n * n < high and n % 2]
        return _map_value(question, len(roots), "odd_squares", f"odd roots are {roots}")

    match = re.search(r"sum of all positive integer values of \$n\$ such that \$n\^2\$ is a factor of \$(\d+)\$", text, re.I)
    if match:
        number = int(match.group(1))
        values = [n for n in range(1, math.isqrt(number) + 1) if number % (n * n) == 0]
        return _map_value(question, sum(values), "square_divisors", f"valid n values are {values}")

    match = re.search(r"sum of all integer solutions to \$\|n\| < \|n-(\d+)\| < (\d+)\$", text, re.I)
    if match:
        shift, bound = map(int, match.groups())
        values = [n for n in range(-10 * bound, 10 * bound + 1) if abs(n) < abs(n - shift) < bound]
        return _map_value(question, sum(values), "integer_inequality", f"integer solutions are {values}")

    match = re.search(r"roll two fair (\d+)-sided dice.*sum to (\d+)", text, re.I)
    if match:
        sides, target = map(int, match.groups())
        outcomes = list(itertools.product(range(1, sides + 1), repeat=2))
        favorable = sum(a + b == target for a, b in outcomes)
        return _map_value(question, Fraction(favorable, len(outcomes)), "dice_sum", f"{favorable}/{len(outcomes)} outcomes")

    match = re.search(r"die marked with the numbers 1 through (\d+).*?(\d+|six)-sided die.*product.*multiple of (\d+)", text, re.I)
    if match:
        first_raw, second_raw, divisor_raw = match.groups()
        first_sides = int(first_raw)
        second_sides = 6 if second_raw.lower() == "six" else int(second_raw)
        divisor = int(divisor_raw)
        outcomes = list(itertools.product(range(1, first_sides + 1), range(1, second_sides + 1)))
        favorable = sum((a * b) % divisor == 0 for a, b in outcomes)
        return _map_value(question, Fraction(favorable, len(outcomes)), "dice_product", f"{favorable}/{len(outcomes)} outcomes")

    match = re.search(r"for some \$a,b,c\$ we have \$a\+b\+c = (-?\d+)\$, \$ab\+ac\+bc = (-?\d+)\$ and \$abc = (-?\d+)\$", text)
    if match:
        sum1, sum2, product = map(int, match.groups())
        result = sum1**3 - 3 * sum1 * sum2 + 3 * product
        return _map_value(question, result, "newton_sum_cubes", f"p3=s1^3-3*s1*s2+3*s3={result}")

    match = re.search(r"If \$\(2x \+ 3y\)\^2 = (-?\d+)\$ and \$xy = (-?\d+)\$", text)
    if match:
        square, xy = map(int, match.groups())
        result = square - 12 * xy
        return _map_value(question, result, "expanded_square", f"4x^2+9y^2={square}-12({xy})={result}")

    match = re.search(r"greatest possible quotient.*set \$\\\{(.+?)\\\}\$", text, re.I)
    if match:
        fractions = [
            Fraction(int(a), int(b))
            for a, b in re.findall(r"\\frac\{(\d+)\}\{(\d+)\}", match.group(1))
        ]
        remainder = re.sub(r"\\frac\{\d+\}\{\d+\}", "", match.group(1))
        fractions.extend(Fraction(int(value)) for value in re.findall(r"\b\d+\b", remainder))
        if len(fractions) < 2:
            return None
        result = max(a / b for a in fractions for b in fractions if a != b)
        return _map_value(question, result, "maximum_quotient", f"max/min = {max(fractions)}/{min(fractions)} = {result}")

    return None
