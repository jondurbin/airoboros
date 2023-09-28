import re
import json
import math
import sympy
import logging
import numpy as np

MACHINE_EPSILON_32 = float(np.finfo(np.float32).eps)


class ParseFailure(ValueError):
    ...


def add(*args):
    return sum([evaluate(val) for val in args])


def multiply(*args):
    result = evaluate(args[0])
    for idx in range(1, len(args)):
        result *= evaluate(args[idx])
    return result


def divide(*args):
    result = evaluate(args[0])
    for idx in range(1, len(args)):
        result = result / evaluate(args[idx])
    return result


def subtract(*args):
    result = evaluate(args[0])
    for idx in range(1, len(args)):
        result -= evaluate(args[idx])
    return result


METHOD_MAP = {
    "divide": divide,
    "multiply": multiply,
    "add": add,
    "subtract": subtract,
    "power": math.pow,
    "square": lambda *x: math.pow(x[0], 2),
    "arccos": math.acos,
    "arctan": math.atan,
    "arcosh": math.acosh,
    "arcoth": lambda *x: math.log((1 + x[0]) / (x[0] + 1)) / 2.0,
    "arctan2": math.atan2,
    "arsech": lambda *x: math.log((1 / x[0]) + math.sqrt((1 / math.pow(x[0], 2)) - 1)),
    "arsinh": math.asinh,
    "artanh": math.atanh,
    "max": max,
    "round": round,
    "logoneplus": math.log1p,
    "signgamma": math.gamma,
    "rational": lambda *x: x[0] / x[1],
    "clamp": lambda *x: x[1] if x[0] < x[1] else x[2] if x[0] > x[2] else x[0],
    "negate": lambda *x: x[0] * -1,
}


def evaluate(obj):
    logging.debug(f"Evaluating: {obj}")

    # Excellent, the value is already a simple int/float type.
    if isinstance(obj, int):
        return float(obj)
    if isinstance(obj, float):
        return obj

    # Ok, slightly more complex, strings can be numbers, symbols, or fractions.
    if isinstance(obj, str):
        if obj == "Infinity":
            return math.inf
        if obj == "-Infinity":
            return math.inf * -1
        if obj == "ExponentialE":
            return math.e
        if obj == "MachineEpsilon":
            return MACHINE_EPSILON_32
        if obj == "CatalanConstant":
            return 0.91596559
        m = re.match(r"^([0-9\.]+)\s*/([0-9\.]+)\s*$", obj)
        if m:
            return float(m.group(1) / m.group(2))
        try:
            value = float(obj)
        except Exception:
            value = getattr(
                math, obj.lower(), getattr(sympy, obj, getattr(sympy.S, obj, None))
            )
        if value is None:
            logging.debug(f"Failed to determine value of {obj}")
            raise ParseFailure(obj)
        return value

    # I guess we can handle some dictionary representations, annoyingly.
    if isinstance(obj, dict):
        if len(obj) > 1:
            raise ParseFailure(f"too lazy to parse complex dicts right now: {obj}")
        key = list(obj)[0]
        value = obj[key]
        if key in ("sym", "num"):
            return evaluate(value)

    # Lists, which should mostly be [operator, value 0, ..., value n]
    if isinstance(obj, list):
        operator = obj[0].lower()
        method = METHOD_MAP.get(
            operator.lower(),
            getattr(math, operator.lower(), getattr(sympy, operator, None)),
        )
        if not method:
            logging.debug(f"Could not determine method from {obj}")
            raise ParseFailure(obj)
        values = [evaluate(item) for item in obj[1:]]
        return method(*values)


def main():
    items = [
        json.loads(line)
        for line in open("combined-solutions-mathjson.jsonl").readlines()
    ]

    for item in items:
        # Find the "correct" answer, if provided.
        expected_answer = None
        m = re.search(r"the answer is: (.*)", item["solution"], re.I)
        if m:
            numeric = re.search(r"([0-9\.]+)", m.group(1))
            if numeric:
                try:
                    expected_answer = float(numeric.group(1))
                except Exception:
                    ...
            if expected_answer is None:
                logging.warning(f"Could not parse {m.group(1)} to float")

        # Perform the MathJSON calculation.
        m = re.search("<mathjson>(.*?)</mathjson>", item["response"], re.I | re.DOTALL)
        if not m:
            logging.warning("Solution did not contain mathjson!")
            continue
        formulation_text = m.group(1)
        try:
            formulation = json.loads(formulation_text)
        except Exception:
            logging.warning(f"MathJSON parse failure: {formulation_text}")
            continue
        try:
            result = evaluate(formulation)
        except Exception as exc:
            logging.warning(f"Error evaluating: {exc}")
            continue

        # Compare expected answer (if provided) to calculated answer.
        if expected_answer is not None:
            try:
                if (
                    int(result) != int(expected_answer)
                    and float(
                        abs(expected_answer - result) / float(expected_answer or result)
                    )
                    > 0.01
                ):
                    logging.error(f"Expected: {expected_answer} Actual  : {result}")
                else:
                    logging.info(f"Validated {expected_answer} vs {result}")
            except Exception as exc:
                logging.warning(f"Validation error: {exc}")
        else:
            logging.info(f"Dangerously assuming this is correct! {result}")


if __name__ == "__main__":
    main()
