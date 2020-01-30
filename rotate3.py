# -*- encoding:utf-8 -*-
import os
import numpy as np
import matplotlib.pylab as plt
import math
import sys
from utils import degreeToRadian, stringToFloat
from numpy import genfromtxt

# dirNames = ['0', '1', '7', '8', '9', '10', '12', '13', '14', '16']
# dirNames = ['virtualRotate']
dirNames = ['0_test']
# workspacePath = "C:\\Users\\three\\Desktop\\workspace\\EIS_lab_intern\\work1\\my_workspace\\rotated_sample\\test\\"
# workspacePath = '/Users/jafffy/workspace/depthmaps/rotation/sample/'
workspacePath = '/Users/jafffy/workspace/depthmaps/rotation/rotate v3/test/realRotate/'

height = 480
width = 640

hFOV = degreeToRadian(56.559)
vFOV = math.atan(height / width * math.tan(hFOV / 2)) * 2

tan_hFOV_half = math.tan(hFOV / 2)
tan_vFOV_half = math.tan(vFOV / 2)

DEPTH_THRSHOLD = 100


# print ("hFOV: " + str(hFOV) + ", vFOV: " + str(vFOV))
# convert 2D depth array to 2D coordinate(x, y, z) array
def depthToCoordinates(arr):
    ret = []

    for i in range(height):
        for j in range(width):
            cd = arr[i][j]

            if cd > DEPTH_THRSHOLD or math.isnan(cd):
                continue

            k_x = tan_hFOV_half * DEPTH_THRSHOLD / (width / 2)
            cx = (width / 2 - j) * k_x

            k_y = tan_vFOV_half * DEPTH_THRSHOLD / (height / 2)
            cy = (height / 2 - i) * k_y

            ret.append([cx, cy, cd, 1.0])

    return ret

inv_width = 1 / width
inv_height = 1 / height

k_x = tan_hFOV_half * DEPTH_THRSHOLD * inv_width * 2
k_y = tan_vFOV_half * DEPTH_THRSHOLD * inv_height * 2

screen_base = np.array([width * 0.5, height * 0.5, 0, 0])

P = np.array([
    [1/k_x, 0, 0, 0],
    [0, 1/k_y, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


# convert 2D coordinate(x, y, z) array to 2D depth array
def coordinates_to_depth(crdArr, vt, fsz):
    ret = plt.zeros((height, width))

    pts = [(0, 0, 0) for _ in range(4 * len(crdArr))]

    import time
    t = time.time()
    crdArr = screen_base - crdArr

    for idx in range(len(crdArr)):
        cc = crdArr[idx][0]
        cr = crdArr[idx][1]
        cd = crdArr[idx][2]

        ccs = [math.floor(cc), math.floor(cc), math.ceil(cc), math.ceil(cc)]
        crs = [math.floor(cr), math.floor(cr), math.ceil(cr), math.ceil(cr)]

        for k in range(len(ccs)):
            tc = ccs[k]
            tr = crs[k]

            if tr < 0 or tc < 0 or tr >= height or tc >= width:
                continue

            pts[4 * idx + k] = (tc, tr, cd)

    t0 = time.time()
    print(t0 - t)

    for point in pts:
        cc, cr, cd = point

        if ret[cr][cc] > 0:
            ret[cr][cc] = max(ret[cr][cc], cd)
        elif ret[cr][cc] == 0:
            ret[cr][cc] = cd

    print(time.time() - t0)

    from scipy import ndimage
    ret = ndimage.median_filter(ret, fsz)

    return ret


# rotate 2D coordinate array with X axis
def get_rotation_matrix_x_axis(degree):
    theta = degreeToRadian(degree)
    c = math.cos(theta)
    s = math.sin(theta)

    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


txtArr = np.zeros((height, width))  # 2D depth array


def makePng(curTxtPath):
    import asyncio

    txtArr = genfromtxt(curTxtPath + '.txt', delimiter=' ')
    crdArr = np.array(depthToCoordinates(txtArr))

    new_origin = (np.min(crdArr, axis=0) + np.max(crdArr, axis=0)) * 0.5

    T = np.array([
        [1.0, 0.0, 0.0, new_origin[0]],
        [0.0, 1.0, 0.0, new_origin[1]],
        [0.0, 0.0, 1.0, new_origin[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    invT = np.array([
        [1.0, 0.0, 0.0, -new_origin[0]],
        [0.0, 1.0, 0.0, -new_origin[1]],
        [0.0, 0.0, 1.0, -new_origin[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    transposed_point_cloud = np.dot(invT, np.transpose(crdArr))

    import time

    # 회전한 png파일생성
    vt = 0
    for fsz in [5]:
        for rotateDegree in range(-15, 15 + 1, 1):
            t = time.time()

            Rx = get_rotation_matrix_x_axis(rotateDegree)
            rotatedCrdArr = np.transpose(np.dot(np.dot(P, np.dot(T, Rx)), transposed_point_cloud))
            rotatedDepthArr = coordinates_to_depth(rotatedCrdArr, vt, fsz)  # rotateDepthArr로 txt파일 만들면 회전변환된 depth 파일

            def save_imagefile(newPngPath, rotatedDepthArr, t):
                plt.imsave(newPngPath, rotatedDepthArr)
                plt.imshow(rotatedDepthArr)
                print(newPngPath + " done within %f" % (time.time() - t))

            save_imagefile(curTxtPath + "-vt_" + str(vt) + "-fsz_" + str(fsz) + "-" + str(rotateDegree) + ".png",
                           rotatedDepthArr, t)

    return 0


def main():
    for curDirName in dirNames:
        txtPath = os.path.join(workspacePath + curDirName, '0')
        makePng(txtPath)


if __name__ == "__main__":
    main()
