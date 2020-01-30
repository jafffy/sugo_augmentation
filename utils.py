import math
import numpy as np


# convert degree unit to radian unit
def degreeToRadian(degree):
    return math.pi / 180 * degree


def stringToFloat(num):
    try:
        num = float(num)
        if np.isnan(num):
            num = 170  # 임의로 넣은 숫자에요. 의미가 크게 x
    except:
        num = 0.0
    return num
