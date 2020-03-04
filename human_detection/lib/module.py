import re
import math
import numpy as np


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def to_quaternion_rad(w, z):
    return math.acos(w) * 2 * np.sign(z)
