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

hFOV = degreeToRadian(56.559)  # TODO: Confirm that the number is for hFOV
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

            k_x = tan_hFOV_half * cd / (width / 2)
            cx = (width / 2 - j) * k_x

            k_y = tan_vFOV_half * cd / (height / 2)
            cy = (height / 2 - i) * k_y

            ret.append([cx, cy, cd, 1.0])

    return ret


# convert 2D coordinate(x, y, z) array to 2D depth array
def coordinatesTodepth(crdArr, fov_variant, fsz):
    hFOV_prime = degreeToRadian(56.559 + fov_variant)  # TODO: Confirm that the number is for hFOV
    vFOV_prime = math.atan(height / width * math.tan(hFOV_prime / 2)) * 2

    tan_hFOV_half_prime = math.tan(hFOV_prime / 2)
    tan_vFOV_half_prime = math.tan(vFOV_prime / 2)

    ret = plt.zeros((height, width))

    for point in crdArr:
        assert point[2] > 0

        cx = point[0]
        cy = point[1]
        cd = point[2]

        k_x = tan_hFOV_half_prime * cd / (width / 2)
        cc = (width / 2 * k_x - cx) / k_x
        k_y = tan_vFOV_half_prime * cd / (height / 2)
        cr = (height / 2 * k_y - cy) / k_y

        ccs = [math.floor(cc), math.floor(cc), math.ceil(cc), math.ceil(cc)]
        crs = [math.floor(cr), math.floor(cr), math.ceil(cr), math.ceil(cr)]

        for k in range(len(ccs)):
            tc = ccs[k]
            tr = crs[k]

            if tr < 0 or tc < 0 or tr >= height or tc >= width:
                continue

            # z값이 음수인 경우에 대해서는 어떻게 처리할지 미정
            if ret[tr][tc] == 0:
                ret[tr][tc] = cd
            else:
                ret[tr][tc] = min(ret[tr][tc], cd)

    for i in range(height):
        for j in range(width):
            if ret[i][j] == 0:
                ltR = max(i - fsz // 2, 0)  # left top row of filter
                ltC = max(j - fsz // 2, 0)  # left top column of filter
                rbR = min(i + fsz // 2, height - 1)
                rbC = min(j + fsz // 2, width - 1)

                # pixelCnt = (rbR-ltR+1)*(rbC-ltC+1)

                cands = []
                for tr in range(ltR, rbR + 1):
                    for tc in range(ltC, rbC + 1):
                        if (tr == i and tc == j):
                            continue
                        cands.append(ret[tr][tc])

                ret[i][j] = cands[len(cands) // 2]  # median

    return ret


# rotate 2D coordinate array with Y axis
def rotateWithYaxis(crdArr, degree, newOrigin):
    ret = [[] for _ in range(height)]

    theta = degreeToRadian(degree)

    c = math.cos(theta)
    s = math.sin(theta)

    Ry = [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    for i in range(height):
        for j in range(width):
            cx = crdArr[i][j][0] - newOrigin[0]
            cy = crdArr[i][j][1] - newOrigin[1]
            cz = crdArr[i][j][2] - newOrigin[2]
            A = [cx, cy, cz, 1.0]
            B = []

            for k in range(4):
                tmp = 0
                for l in range(4):
                    tmp += Ry[k][l] * A[l]
                B.append(tmp)
            nx = B[0] + newOrigin[0]
            ny = B[1] + newOrigin[1]
            nz = B[2] + newOrigin[2]

            ret[i].append([nx, ny, nz])
    return ret


# rotate 2D coordinate array with X axis
def rotateWithXaxis(crdArr, degree, newOrigin):
    theta = degreeToRadian(degree)
    c = math.cos(theta)
    s = math.sin(theta)

    Rx = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    T = np.array([
        [1.0, 0.0, 0.0, newOrigin[0]],
        [0.0, 1.0, 0.0, newOrigin[1]],
        [0.0, 0.0, 1.0, newOrigin[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    invT = np.array([
        [1.0, 0.0, 0.0, -newOrigin[0]],
        [0.0, 1.0, 0.0, -newOrigin[1]],
        [0.0, 0.0, 1.0, -newOrigin[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return np.transpose(
        np.dot(T,
               np.dot(Rx,
                      np.dot(invT,
                          np.transpose(crdArr)))))


txtArr = np.zeros((height, width))  # 2D depth array


def makePng(curTxtPath):
    import asyncio

    txtArr = genfromtxt(curTxtPath + '.txt', delimiter=' ')
    crdArr = np.array(depthToCoordinates(txtArr))

    # 회전한 png파일생성
    vt = 0
    for fsz in range(3, 4, 2):
        for fov_variant in range(-15, 15 + 1, 1):
            reprojected_depth_image = coordinatesTodepth(crdArr, fov_variant, fsz)  # rotateDepthArr로 txt파일 만들면 회전변환된 depth 파일

            async def save_imagefile(newPngPath, reprojected_depth_image):
                plt.imsave(newPngPath, reprojected_depth_image)
                plt.imshow(reprojected_depth_image)
                print(newPngPath + " done")

            asyncio.run(
                save_imagefile(curTxtPath + "-fov_variant_" + str(fsz) + "-" + str(fov_variant) + ".png",
                               np.array(reprojected_depth_image, copy=True)))

    return 0


def main():
    for curDirName in dirNames:
        txtPath = os.path.join(workspacePath + curDirName, '0')
        makePng(txtPath)


if __name__ == "__main__":
    main()
