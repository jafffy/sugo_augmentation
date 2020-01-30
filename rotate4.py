#-*- encoding:utf-8 -*-
import os
import numpy as np
import matplotlib.pylab as plt
import math
import sys
import time
from utils import degreeToRadian, stringToFloat

# dirNames = ['0', '1', '7', '8', '9', '10', '12', '13', '14', '16']
# dirNames = ['13']
dirNames = ['virtualRotate']
fileNames = ['0']

# workspacePath = "C:\\Users\\three\\Desktop\\workspace\\EIS_lab_intern\\work1\\my_workspace\\new_sample\\"
workspacePath = "C:\\Users\\three\\Desktop\\workspace\\EIS_lab_intern\\work1\\my_workspace\\rotated_sample\\test\\"

height = 480
width = 640

hFOV = degreeToRadian(56.559)
vFOV = math.atan(height / width * math.tan(hFOV / 2)) * 2

tan_hFOV_half = math.tan(hFOV / 2)
tan_vFOV_half = math.tan(vFOV / 2)

DEPTH_THRSHOLD = 100


# convert 2D depth array to 2D coordinate(x, y, z) array
def depth_to_coordinates(arr):
    ret = []

    for i in range(height):
        for j in range(width):
            cd = arr[i][j]
            k_x = tan_hFOV_half * cd / (width // 2)
            cx = (width/2 - j)*k_x

            k_y = tan_vFOV_half * cd / (height // 2)
            cy = (height/2 - i)*k_y

            ret.append([cx, cy, cd, 1.0])

    return ret


# convert 2D coordinate(x, y, z) array to 2D depth array
def coordinatesTodepth(crdArr, vt, fsz):
    ret = plt.zeros((height, width))

    t = time.time()
    for index in range(len(crdArr)):
        cx = crdArr[index][0]
        cy = crdArr[index][1]
        cz = crdArr[index][2]

        if(cz==0):
            continue

        k_x = tan_hFOV_half*cz / (width//2)
        k_y = tan_vFOV_half*cz / (height//2)

        cc = ((width//2)*k_x - cx)/k_x
        cr = ((height//2)*k_y - cy)/k_y
        cd = math.sqrt(cx*cx+cy*cy+cz*cz)

        ccs = [math.floor(cc), math.floor(cc), math.ceil(cc), math.ceil(cc)]
        crs = [math.floor(cr), math.ceil(cr), math.floor(cr), math.ceil(cr)]

        for k in range(len(ccs)):
            tc = ccs[k]
            tr = crs[k]

            if tr < 0 or tc < 0 or tr >= height or tc >= width:
                continue

            # z값이 음수인 경우에 대해서는 어떻게 처리할지 미정
            if (ret[tr][tc] == 0):
                ret[tr][tc] = cd
            elif (ret[tr][tc]>0):
                ret[tr][tc] = min(ret[tr][tc], cd)

    t0 = time.time()
    print(t0 - t)

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


def make_png(curTxtPath):
    depthTxt = open(curTxtPath + ".txt", "r")
    tmpArr = depthTxt.read().replace('\n', ' ').replace('  ', ' ').split(' ')
    print(len(tmpArr))

    txtArr = plt.zeros((height, width))  # 2D depth array
    index = 0
    for i in range(height):
        for j in range(width):
            txtArr[i][j] = stringToFloat(tmpArr[index])
            if(txtArr[i][j]>DEPTH_THRSHOLD):
                txtArr[i][j] = 0
            index = index + 1

    crdArr = depth_to_coordinates(txtArr)
    newOrigin = (np.min(crdArr, axis=0) + np.max(crdArr, axis=0)) * 0.5   # error occurs
    # newOrigin = [crdArr[height//2*width + width//2][0], crdArr[height//2*width + width//2][1], crdArr[height//2*width + width//2][2]]

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

    transposed_point_cloud = np.dot(invT, np.transpose(crdArr))

    # 회전한 png파일생성
    vt = 0
    for fsz in [5]:
        for rotateDegree in range(-15, 15 + 1, 1):
            t = time.time()
            Rx = get_rotation_matrix_x_axis(rotateDegree)

            rotatedCrdArr = np.transpose(np.dot(np.dot(T, Rx), transposed_point_cloud))
            rotatedDepthArr = coordinatesTodepth(rotatedCrdArr, vt, fsz)  # rotateDepthArr로 txt파일 만들면 회전변환된 depth 파일

            def save_imagefile(newPngPath, rotatedDepthArr, t):
                plt.imsave(newPngPath, rotatedDepthArr)
                plt.imshow(rotatedDepthArr)
                print(newPngPath + " done within %f" % (time.time() - t))

            if (rotateDegree < 0):
                save_imagefile(curTxtPath + "-vt_" + str(vt) + "-fsz_" + str(fsz) + "-m" + str(rotateDegree) + ".png",
                               rotatedDepthArr, t)
            else:
                save_imagefile(curTxtPath + "-vt_" + str(vt) + "-fsz_" + str(fsz) + "-" + str(rotateDegree) + ".png",
                               rotatedDepthArr, t)

    return 0

def main():
    for curDirName in dirNames:
        for curFileName in fileNames:
            txtPath = os.path.join(workspacePath + curDirName, curFileName)
            make_png(txtPath)

if __name__ == "__main__":
    # os.sys.setrecursionlimit(640*480)
    main()
