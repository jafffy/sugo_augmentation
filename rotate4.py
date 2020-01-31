#-*- encoding:utf-8 -*-
import os
import numpy as np
import matplotlib.pylab as plt
import math
import sys
import time
from utils import degreeToRadian, stringToFloat

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
def coordinates_to_depth(crdArr, vt, fsz, theta):
    ret = plt.zeros((height, width))

    print(crdArr)

    t = time.time()
    for index in range(len(crdArr)):
        cx = crdArr[index][0]
        cy = crdArr[index][1]
        cz = crdArr[index][2]

        if(cz==0):
            continue

        hFOV_prime = degreeToRadian(56.559 + theta)
        vFOV_prime = math.atan(height / width * math.tan(hFOV_prime / 2)) * 2

        tan_hFOV_half_prime = math.tan(hFOV_prime / 2)
        tan_vFOV_half_prime = math.tan(vFOV_prime / 2)

        k_x = tan_hFOV_half_prime*cz / (width//2)
        k_y = tan_vFOV_half_prime*cz / (height//2)

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

    from skimage.transform import resize
    ret = resize(ret, (224, 224))

    t0 = time.time()
    print(t0 - t)

    from scipy import ndimage
    ret = ndimage.median_filter(ret, fsz)

    from sklearn.preprocessing import minmax_scale
    ret = minmax_scale(ret.ravel(), feature_range=(0, 1)).reshape(ret.shape)

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


def make_png(curTxtPath, is_np_array=False):
    from pathlib import Path

    path = Path(curTxtPath)
    parent_path = str(path.parent)

    for rotateDegree in [-8, -4, -2, 0, 2, 4, 8]:
        for FoVDegree in [0]:
            for scale_degree in range(3):
                dirname = str(rotateDegree) + '_' + str(FoVDegree) + '_' + str(1.0 + scale_degree * 0.1)
                dirpath = os.path.join(parent_path, dirname)

                if not os.path.exists(dirpath):
                    os.mkdir(dirpath)

    if is_np_array:
        tmpArr = np.load(curTxtPath)
    else:
        depthTxt = open(curTxtPath, "r")
        tmpArr = depthTxt.read().replace('\n', ' ').replace('  ', ' ').split(' ')
    print(len(tmpArr))

    txtArr = plt.zeros((height, width))  # 2D depth array

    for i in range(height):
        for j in range(width):
            txtArr[i][j] = stringToFloat(tmpArr[i][j])

    crdArr = depth_to_coordinates(txtArr)
    newOrigin = (np.min(crdArr, axis=0) + np.max(crdArr, axis=0)) * 0.5   # error occurs

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
        for rotateDegree in [-8, -4, -2, 0, 2, 4, 8]:
            for FoVDegree in [0]: # range(-4, 4 + 1, 2):
                for scale_degree in range(3):
                    t = time.time()
                    Rx = get_rotation_matrix_x_axis(rotateDegree)

                    scale_factor = 1.0 + scale_degree * 0.1

                    S = np.array([
                        [scale_factor, 0.0, 0.0, 0.0],
                        [0.0, scale_factor, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ])

                    rotatedCrdArr = np.transpose(np.dot(np.dot(T, np.dot(Rx, S)), transposed_point_cloud))
                    rotatedDepthArr = coordinates_to_depth(rotatedCrdArr, vt, fsz, np.sign(FoVDegree) * (2 ** abs(FoVDegree)))  # rotateDepthArr로 txt파일 만들면 회전변환된 depth 파일

                    def save_imagefile(newPngPath, rotatedDepthArr, t):
                        plt.imsave(newPngPath, rotatedDepthArr)
                        plt.imshow(rotatedDepthArr)
                        print(newPngPath + " done within %f" % (time.time() - t))

                    def save_npyfile(npy_path, rotatedDepthArr, t):
                        np.save(npy_path, rotatedDepthArr)
                        print(npy_path + " done within %f" % (time.time() - t))

                    dirname = str(rotateDegree) + '_' + str(FoVDegree) + '_' + str(scale_factor)

                    dirname = str(rotateDegree) + '_' + str(FoVDegree) + '_' + str(scale_factor)
                    dirpath = os.path.join(parent_path, dirname)
                    filename = os.path.basename(curTxtPath).split('.')[0] + '.png'
                    
                    # save_imagefile(curTxtPath + "-vt_" + str(vt) + "-fsz_" + str(fsz) + "-" + str(rotateDegree) + "-fov" + str(np.sign(FoVDegree) * (2 ** abs(FoVDegree))) + "-scale" + str(scale_degree * 0.1) + ".png", rotatedDepthArr, t)
                    save_imagefile(os.path.join(dirpath, filename), rotatedDepthArr, t)
                    # save_npyfile(os.path.join(dirpath, filename), rotatedDepthArr, t)

def main():
    import os

    assert len(sys.argv) == 2

    if sys.argv[1].endswith('.txt'):
        with open(sys.argv[1], 'r') as f:
            for line in f:
                filename = line.strip()

                if not os.path.isfile(filename): 
                    continue
                
                make_png(filename, is_np_array=True)
        return

    base_dir = sys.argv[1]

    def flatten_dir(base_dir, depth):
        if depth == 1:
            return [os.path.join(base_dir, x) for x in os.listdir(base_dir)]
        else:
            flatten_dirs = []

            for path in os.listdir(base_dir):
                flatten_dirs += flatten_dir(os.path.join(base_dir, path), depth - 1)

            return flatten_dirs

    for text_path in filter(lambda x: x.endswith('txt'), flatten_dir(base_dir, 3)):
        make_png(text_path)
        
if __name__ == "__main__":
    # os.sys.setrecursionlimit(640*480)
    main()
