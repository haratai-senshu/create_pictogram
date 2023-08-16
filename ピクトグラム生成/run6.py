import argparse
import sys
import time
import re
import os
import math
import glob
import moviepy.editor as mp
from moviepy.editor import *
from PIL import Image
import subprocess
import threading
import shutil
import psutil

from tf_pose import common
import cv2 as cv
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


pathall = os.path.dirname(os.path.abspath(__file__))


def proc(num,num2,n1_1) :
    predict_time = []
    i11 = 0
    ji = []
    ir1 = 0
    for n1 in range(num,num2):

        ji1 = time.time()
        try :
            imkaku = str(pathall) + "/image_first/" + str(n1) + ".jpg"
            im=Image.open(str(imkaku))
            #time_1 = time.time()
            command = ["python",str(pathall) + "/" + "videopo.py",str(n1)]
            proc = subprocess.Popen(command)
            global th_t1
            th_t1 = time.time()
            print("呼び出し中")
            proc.communicate()
        except Exception as e :
            print(e)


        ji2 = time.time()
        ji3 = float(ji2) - float(ji1)
        ji.append(ji3)
        ir1 = 0
        for ir in ji :
            ir1 += float(ir)
        ir1 = float(ir1) / float(len(ji))

        global ir2
        ir2 = ir1 / 2
    global g1
    g1 = 1

def proc2(num,num2) :
    th_t_list = []
    for n1 in range(num,num2):
        try :
            th_t1 = time.time()
            while True:
                try :
                    if g1 != 1 :
                        pass
                    else :
                        break
                except :
                    pass
                try :
                    cpu = psutil.cpu_percent()
                    if cpu >= 80 :
                        if ir2 :
                            break
                        else :
                            pass
                    time.sleep(0.1)
                except :
                    pass

            time.sleep(ir2/1.5)
            imkaku = str(pathall) + "/image_first/" + str(n1) + ".jpg"
            im=Image.open(str(imkaku))
            time_1 = time.time()
            command = ["python",str(pathall) + "/" + "videopo.py",str(n1)]
            proc = subprocess.Popen(command)
            print("呼び出し中2")
            proc.communicate()

            th_t2 = time.time()
            th_t3 = th_t2 - th_t1
            th_t_list.append(th_t3)
            th_t4 = 0
            for i in th_t_list :
                th_t4 += i
            th_t4 = th_t4 / len(th_t_list)
            num3 = num2 - n1
            th_t = float(num3) * float(th_t4)
            end_time = time.strftime('%d:%H:%M:%S', time.gmtime(math.floor(th_t)))

            print('あと、' , str(end_time) , 'かかると予想されます。')

        except Exception as e :
            print(e)

while True :
    path_video_1 = sys.argv[1]

    if path_video_1 == "end" :
        sys.exit()
    elif path_video_1 == "remove-completed-video" :
        try :
            os.remove(str(pathall) + '/EX_video/pictogram.mp4')
            print("正常に削除できました。")
        except :
            print("ファイルを削除できません。ファイルが存在しないと思われます。")


    else :
        try:
            pass
            im=Image.open(str(path_video_1))
            print("画像ファイルです。")
            parser = argparse.ArgumentParser(description='tf-pose-estimation run')
            #parser.add_argument('--image', type=str, default='./images/p1.jpg')
            #parser.add_argument('--model', type=str, default='cmu',
                                #help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
            #parser.add_argument('--resize', type=str, default='432x368',
                                #help='if provided, resize images before they are processed. '
                                     #'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
            #parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                                #help='if provided, resize heatmaps before they are post-processed. default=1.0')

            #args = parser.parse_args()
            #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

            w, h = model_wh('432x368')
            if w == 0 or h == 0:
                e = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368))
            else:
                e = TfPoseEstimator(get_graph_path('cmu'), target_size=(w, h))

            # estimate human poses from a single image !
            image = common.read_imgfile(str(path_video_1), None, None)
            if image is None:
                logger.error('Image can not be read, path=%s' % str(path_video_1))
                sys.exit(-1)

            #humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

        #python run2.py --model=mobilenet_thin --resize=432x368 --image=./jon.png
            # 対象画像読み込み
            img = cv.imread(str(path_video_1),cv.IMREAD_COLOR)
            # 画像の大きさを取得
            height, width = img.shape[:2]

            # 指定サイズと色で背景画像を生成
            img = np.full((height, width, 3), 255, dtype=np.uint8)

            for human in humans :
                parts = human.body_parts

                for n in parts.keys() :
                    part = re.findall("\d+\.\d+", str(parts[n]))
                    if n == 0 :
                        頭 = part
                    elif n == 1 :
                        胸 = part
                    elif n == 2 :
                        右肩 = part
                    elif n == 3 :
                        右肘 = part
                    elif n == 4 :
                        右手 = part
                    elif n == 5 :
                        左肩 = part
                    elif n == 6 :
                        左肘 = part
                    elif n == 7 :
                        左手 = part
                    elif n == 8 :
                        右股関節 = part
                    elif n == 9 :
                        右膝 = part
                    elif n == 10 :
                        右足 = part
                    elif n == 11 :
                        左股関節 = part
                    elif n == 12 :
                        左膝 = part
                    elif n == 13 :
                        左足 = part
                    elif n == 15 :
                        右目 = part
                    elif n == 17 :
                        目 = part


        #python run2.py --model=mobilenet_thin --resize=432x368 --image=/Users/tyakyumyou/ildoonet-tf-pose-estimation/image_first/0.jpg
                #右肩
                try:
                    a = np.array([math.floor(float(右肩[0])*width), math.floor(float(右肩[1])*height)])
                    b = np.array([math.floor(float(右肘[0])*width), math.floor(float(右肘[1])*height)])
                    long_mhi = math.floor(float(np.linalg.norm(a-b))/4.5) #長さ
                    long_mhi_1 = math.floor(float(np.linalg.norm(a-b)))

                    if float(width) >= float(height) :#横長の場合
                        if long_mhi >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右肩[0])*width), math.floor(float(右肩[1])*height)), long_mhi, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long_mhi >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右肩[0])*width), math.floor(float(右肩[1])*height)), long_mhi, (100, 33, 2), thickness=-1)


                except Exception as e:
                    print(e)
                #右肘
                try:
                    long = math.floor(float(np.linalg.norm(a-b))/4.5) #長さ
                    long1 = math.floor(float(np.linalg.norm(a-b))/4.5/2)
                    long = math.floor((long + long1) / 2)
                    if float(width) >= float(height) :#横長の場合
                        if long >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右肘[0])*width), math.floor(float(右肘[1])*height)), long, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右肘[0])*width), math.floor(float(右肘[1])*height)), long, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)
                #右肘右肩、右上腕
                try :
                    #右肩側の座標
                    # 円の中点
                    p = (math.floor(float(右肩[0])*width), math.floor(float(右肩[1])*height))
                    # 半径r[px]
                    r = long_mhi
        #python run2.py --model=mobilenet_thin --resize=432x368 --image=./jon2.jpg
                    vec = b - a
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t = t.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t2 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t2 = t2.tolist()
                    #右肘側の座標
                    # 円の中点
                    p = (math.floor(float(右肘[0])*width), math.floor(float(右肘[1])*height))
                    # 半径r[px]
                    r = long

                    vec = a - b
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t3 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t3 = t3.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t4 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t4 = t4.tolist()
                    points = np.array([(math.floor(t[0]), math.floor(t[1])), (math.floor(t2[0]), math.floor(t2[1])), (math.floor(t3[0]), math.floor(t3[1])), (math.floor(t4[0]), math.floor(t4[1]))])
                    if float(height) >= float(width) :
                        if float(long_mhi_1) >= float(width*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                    else :
                        if float(long_mhi_1) >= float(height*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                except Exception as e :
                    print(e)


        #python run2.py --model=mobilenet_thin --resize=432x368 --image=./jon2.jpg
                #右手
                try:
                    long_mite = math.floor(float(np.linalg.norm(a-b))/4.5/2) #長さ
                    if float(width) >= float(height) :#横長の場合
                        if long_mite >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右手[0])*width), math.floor(float(右手[1])*height)), long_mite, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long_mite >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右手[0])*width), math.floor(float(右手[1])*height)), long_mite, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)

                #右腕
                try :
                    #右肘側の座標
                    a = np.array([math.floor(float(右肘[0])*width), math.floor(float(右肘[1])*height)])
                    b = np.array([math.floor(float(右手[0])*width), math.floor(float(右手[1])*height)])
                    # 円の中点
                    p = (math.floor(float(右肘[0])*width), math.floor(float(右肘[1])*height))
                    # 半径r[px]
                    r = long

                    vec = b - a
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t = t.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t2 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t2 = t2.tolist()
                    #右手側の座標
                    # 円の中点
                    p = (math.floor(float(右手[0])*width), math.floor(float(右手[1])*height))
                    # 半径r[px]
                    r = long_mite

                    vec = a - b
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t3 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t3 = t3.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t4 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t4 = t4.tolist()
                    points = np.array([(math.floor(t[0]), math.floor(t[1])), (math.floor(t2[0]), math.floor(t2[1])), (math.floor(t3[0]), math.floor(t3[1])), (math.floor(t4[0]), math.floor(t4[1]))])

                    a1 = np.array([math.floor(float(右肘[0])*width), math.floor(float(右肘[1])*height)])
                    b1 = np.array([math.floor(float(右手[0])*width), math.floor(float(右手[1])*height)])
                    long_mi_ude = math.floor(float(np.linalg.norm(a1-b1))) #長さ
                    if float(height) >= float(width) :
                        if float(long_mi_ude) >= float(width*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                    else :
                        if float(long_mi_ude) >= float(height*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                except Exception as e :
                    print(e)






                #左肩
                try:
                    a = np.array([math.floor(float(左肩[0])*width), math.floor(float(左肩[1])*height)])
                    b = np.array([math.floor(float(左肘[0])*width), math.floor(float(左肘[1])*height)])
                    long_hika = math.floor(float(np.linalg.norm(a-b))/4.5) #長さ
                    long_hika_1 = math.floor(float(np.linalg.norm(a-b)))
                    if float(width) >= float(height) :#横長の場合
                        if long_hika >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左肩[0])*width), math.floor(float(左肩[1])*height)), long_hika, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long_hika >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左肩[0])*width), math.floor(float(左肩[1])*height)), long_hika, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)
                #左肘
                try:
                    long = math.floor(float(np.linalg.norm(a-b))/4.5) #長さ
                    long1 = math.floor(float(np.linalg.norm(a-b))/4.5/2)
                    long = math.floor((long + long1) / 2)
                    if float(width) >= float(height) :#横長の場合
                        if long >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左肘[0])*width), math.floor(float(左肘[1])*height)), long, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左肘[0])*width), math.floor(float(左肘[1])*height)), long, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)

                #左腕、左上腕
                #右肘右肩、右上腕
                try :
                    #左肩側の座標
                    # 円の中点
                    p = (math.floor(float(左肩[0])*width), math.floor(float(左肩[1])*height))
                    # 半径r[px]
                    r = long_hika

                    vec = b - a
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t = t.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t2 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t2 = t2.tolist()
                    #右肘側の座標
                    # 円の中点
                    p = (math.floor(float(左肘[0])*width), math.floor(float(左肘[1])*height))
                    # 半径r[px]
                    r = long

                    vec = a - b
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t3 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t3 = t3.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t4 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t4 = t4.tolist()
                    points = np.array([(math.floor(t[0]), math.floor(t[1])), (math.floor(t2[0]), math.floor(t2[1])), (math.floor(t3[0]), math.floor(t3[1])), (math.floor(t4[0]), math.floor(t4[1]))])


                    if float(height) >= float(width) :
                        if float(long_hika_1) >= float(width*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                    else :
                        if float(long_hika_1) >= float(height*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                except Exception as e :
                    print(e)
        #python run2.py --model=mobilenet_thin --resize=432x368 --image=./jon2.jpg
                #左手
                try:
                    long_hite = math.floor(float(np.linalg.norm(a-b))/4.5/2) #長さ
                    if float(width) >= float(height) :#横長の場合
                        if long_hite >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左手[0])*width), math.floor(float(左手[1])*height)), long_hite, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long_hite >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左手[0])*width), math.floor(float(左手[1])*height)), long_hite, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)

                #左腕
                try :
                    #左肘側の座標
                    a = np.array([math.floor(float(左肘[0])*width), math.floor(float(左肘[1])*height)])
                    b = np.array([math.floor(float(左手[0])*width), math.floor(float(左手[1])*height)])
                    # 円の中点
                    p = (math.floor(float(左肘[0])*width), math.floor(float(左肘[1])*height))
                    # 半径r[px]
                    r = long

                    vec = b - a
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t = t.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t2 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t2 = t2.tolist()
                    #左手側の座標
                    # 円の中点
                    p = (math.floor(float(左手[0])*width), math.floor(float(左手[1])*height))
                    # 半径r[px]
                    r = long_hite

                    vec = a - b
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t3 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t3 = t3.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t4 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t4 = t4.tolist()
                    points = np.array([(math.floor(t[0]), math.floor(t[1])), (math.floor(t2[0]), math.floor(t2[1])), (math.floor(t3[0]), math.floor(t3[1])), (math.floor(t4[0]), math.floor(t4[1]))])

                    a1 = np.array([math.floor(float(左肘[0])*width), math.floor(float(左肘[1])*height)])
                    b1 = np.array([math.floor(float(左手[0])*width), math.floor(float(左手[1])*height)])
                    long_hi_ude = math.floor(float(np.linalg.norm(a1-b1))) #長さ
                    if float(height) >= float(width) :
                        if float(long_hi_ude) >= float(width*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                    else :
                        if float(long_hi_ude) >= float(height*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                except Exception as e :
                    print(e)



                #腰
                try:
                    a = np.array([math.floor(float(右股関節[0])*width), math.floor(float(右股関節[1])*height)])
                    b = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                    long_miko = math.floor(float(np.linalg.norm(a-b))/5) #長さ
                    long_miko_1 = math.floor(float(np.linalg.norm(a-b)))
                    x = math.floor(float(右股関節[0])*width) + math.floor((math.floor(float(左股関節[0])*width) - math.floor(float(右股関節[0])*width)) / 2)
                    y = math.floor((math.floor(float(右股関節[1])*height) + math.floor(float(左股関節[1])*height)) / 2)
                    if float(width) >= float(height) :#横長の場合
                        if long_miko >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, ( x , y ), long_miko, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long_miko >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, ( x , y ), long_miko, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)
                #右膝
                try:
                    long1 = math.floor(float(np.linalg.norm(a-b))/5/2)
                    long = math.floor((long_miko + long1) / 2)
                    if float(width) >= float(height) :#横長の場合
                        if long >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)), long, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)), long, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)

                #右足、太もも
                #右腰右膝、ふともも
                try :
                    #右腰側の座標
                    # 円の中点
                    #python run2.py --model=mobilenet_thin --resize=432x368 --image=./jon2.jpg
                    #a = np.array([math.floor(float(右股関節[0])*width), math.floor(float(右股関節[1])*height)])
                    a = np.array([x,y])
                    b = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                    p = (x, y)
                    # 半径r[px]
                    r = long_miko

                    vec = b - a
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t = t.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t2 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t2 = t2.tolist()
                    #右膝側の座標
                    # 円の中点
                    p = (math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height))
                    # 半径r[px]
                    r = long

                    vec = a - b
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t3 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t3 = t3.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t4 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t4 = t4.tolist()
                    points = np.array([(math.floor(t[0]), math.floor(t[1])), (math.floor(t2[0]), math.floor(t2[1])), (math.floor(t3[0]), math.floor(t3[1])), (math.floor(t4[0]), math.floor(t4[1]))])
                    if float(height) >= float(width) :
                        if float(long_miko_1) >= float(width*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                    else :
                        if float(long_miko_1) >= float(height*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                except Exception as e :
                    print(e)


                #右足
                try:
                    long_mia = math.floor(float(np.linalg.norm(a-b))/5/2) #長さ
                    if float(width) >= float(height) :#横長の場合
                        if long_mia >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右足[0])*width), math.floor(float(右足[1])*height)), long_mia, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long_mia >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(右足[0])*width), math.floor(float(右足[1])*height)), long_mia, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)

                #右足、脚
                #右膝右足、脚
                try :
                    #右膝側の座標
                    a = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                    b = np.array([math.floor(float(右足[0])*width), math.floor(float(右足[1])*height)])
                    # 円の中点
                    p = (math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height))
                    # 半径r[px]
                    r = long

                    vec = b - a
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t = t.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t2 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t2 = t2.tolist()
                    #右肘側の座標
                    # 円の中点
                    p = (math.floor(float(右足[0])*width), math.floor(float(右足[1])*height))
                    # 半径r[px]
                    r = long_mia

                    vec = a - b
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t3 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t3 = t3.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t4 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t4 = t4.tolist()
                    points = np.array([(math.floor(t[0]), math.floor(t[1])), (math.floor(t2[0]), math.floor(t2[1])), (math.floor(t3[0]), math.floor(t3[1])), (math.floor(t4[0]), math.floor(t4[1]))])
                    a1 = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                    b1 = np.array([math.floor(float(右足[0])*width), math.floor(float(右足[1])*height)])
                    long_mi_a = math.floor(float(np.linalg.norm(a1-b1))) #長さ
                    if float(height) >= float(width) :
                        if float(long_mi_a) >= float(width*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                    else :
                        if float(long_mi_a) >= float(height*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))

                except Exception as e :
                    print(e)


        #python run2.py --model=mobilenet_thin --resize=432x368 --image=./jon2.jpg

                #腰2
                try:
                    a = np.array([math.floor(float(左股関節[0])*width), math.floor(float(左股関節[1])*height)])
                    b = np.array([math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)])
                    long_hiko = math.floor(float(np.linalg.norm(a-b))/5) #長さ
                    long_hiko_1 = math.floor(float(np.linalg.norm(a-b)))
                    x = math.floor(float(左股関節[0])*width) + math.floor((math.floor(float(右股関節[0])*width) - math.floor(float(左股関節[0])*width)) / 2)
                    y = math.floor((math.floor(float(左股関節[1])*height) + math.floor(float(右股関節[1])*height)) / 2)
                    if float(width) >= float(height) :#横長の場合
                        if long_hiko >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, ( x , y ), long_hiko, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long_hiko >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, ( x , y ), long_hiko, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)
                #右膝
                try:
                    long1 = math.floor(float(np.linalg.norm(a-b))/5/2)
                    long = math.floor((long_hiko + long1) / 2)
                    if float(width) >= float(height) :#横長の場合
                        if long >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)), long, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)), long, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)

                #左足、太もも
                #左腰左膝、ふともも
                try :
                    #右腰側の座標
                    # 円の中点
                    #python run2.py --model=mobilenet_thin --resize=432x368 --image=./jon2.jpg
                    #a = np.array([math.floor(float(右股関節[0])*width), math.floor(float(右股関節[1])*height)])
                    a = np.array([x,y])
                    b = np.array([math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)])
                    p = (x, y)
                    # 半径r[px]
                    r = long_hiko

                    vec = b - a
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t = t.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t2 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t2 = t2.tolist()
                    #右膝側の座標
                    # 円の中点
                    p = (math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height))
                    # 半径r[px]
                    r = long

                    vec = a - b
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t3 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t3 = t3.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t4 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t4 = t4.tolist()
                    points = np.array([(math.floor(t[0]), math.floor(t[1])), (math.floor(t2[0]), math.floor(t2[1])), (math.floor(t3[0]), math.floor(t3[1])), (math.floor(t4[0]), math.floor(t4[1]))])
                    if float(height) >= float(width) :
                        if float(long_hiko_1) >= float(width*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))

                    else :
                        if float(long_hiko_1) >= float(height*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))
                except Exception as e :
                    print(e)


                #左足
                try:
                    long_hia = math.floor(float(np.linalg.norm(a-b))/5/2) #長さ
                    if float(width) >= float(height) :#横長の場合
                        if long_hia >= float(height*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左足[0])*width), math.floor(float(左足[1])*height)), long_hia, (100, 33, 2), thickness=-1)
                    else :#縦長の場合
                        if long_hia >= float(width*0.15) :
                            pass
                        else :
                            cv.circle(img, (math.floor(float(左足[0])*width), math.floor(float(左足[1])*height)), long_hia, (100, 33, 2), thickness=-1)

                except Exception as e:
                    print(e)

                #左足、脚
                #左膝左足、脚
                try :
                    #左膝側の座標
                    a = np.array([math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)])
                    b = np.array([math.floor(float(左足[0])*width), math.floor(float(左足[1])*height)])
                    # 円の中点
                    p = (math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height))
                    # 半径r[px]
                    r = long

                    vec = b - a
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t = t.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t2 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t2 = t2.tolist()
                    #右肘側の座標
                    # 円の中点
                    p = (math.floor(float(左足[0])*width), math.floor(float(左足[1])*height))
                    # 半径r[px]
                    r = long_hia

                    vec = a - b
                    rad = np.arctan2(vec[1], vec[0]) #角度
                    deg = np.rad2deg(rad) #ラジアン → 度
                    # 角度θ[°]
                    theta = deg + 90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t3 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t3 = t3.tolist()
                    # 角度θ[°]
                    theta = deg + -90
                    # degree → rad に変換
                    rad = np.deg2rad(theta)
                    # 移動量を算出
                    rsinθ = r * np.sin(rad)
                    rcosθ = r * np.cos(rad)
                    # 円周上の座標
                    t4 = np.array([p[0] + rcosθ, p[1] + rsinθ])
                    t4 = t4.tolist()
                    points = np.array([(math.floor(t[0]), math.floor(t[1])), (math.floor(t2[0]), math.floor(t2[1])), (math.floor(t3[0]), math.floor(t3[1])), (math.floor(t4[0]), math.floor(t4[1]))])
                    a1 = np.array([math.floor(float(左肘[0])*width), math.floor(float(左肘[1])*height)])
                    b1 = np.array([math.floor(float(左手[0])*width), math.floor(float(左手[1])*height)])
                    long_hi_a = math.floor(float(np.linalg.norm(a1-b1))) #長さ
                    if float(height) >= float(width) :
                        if float(long_hi_a) >= float(width*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))

                    else :
                        if float(long_hi_a) >= float(height*0.4) :
                            pass
                        else :
                            cv.fillConvexPoly(img, points, (100, 33, 2))

                except Exception as e :
                    print(e)

                #頭
                try :
                    try :
                        a = np.array([math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)])
                        b = np.array([math.floor(float(胸[0])*width), math.floor(float(胸[1])*height)])
                        long = math.floor(float(np.linalg.norm(a-b))*0.6) #長さ
                        if float(height) >= float(width) :
                            if float(long) >= float(width*0.2) :
                                pass
                            else :
                                try:
                                    a1 = np.array([math.floor(float(左股関節[0])*width), math.floor(float(左股関節[1])*height)])
                                    b1 = np.array([math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)])
                                    long_hi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                    if long >= long_hi_ca :
                                        pass
                                    else :
                                        cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                except Exception as e :
                                    print(e)
                                    try :
                                        a1 = np.array([math.floor(float(右股関節[0])*width), math.floor(float(右股関節[1])*height)])
                                        b1 = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                                        long_mi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                        if long >= long_mi_ca :
                                            pass
                                        else :
                                            cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                    except Exception as e:
                                        print(e)

                        else :
                            if float(long) >= float(height*0.2) :
                                pass
                            else :
                                try:
                                    a1 = np.array([math.floor(float(左股関節[0])*width), math.floor(float(左股関節[1])*height)])
                                    b1 = np.array([math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)])
                                    long_hi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                    if long >= long_hi_ca :
                                        pass
                                    else :
                                        cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                except Exception as e :
                                    print(e)
                                    try :
                                        a1 = np.array([math.floor(float(右股関節[0])*width), math.floor(float(右股関節[1])*height)])
                                        b1 = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                                        long_mi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                        if long >= long_mi_ca :
                                            pass
                                        else :
                                            cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                    except Exception as e:
                                        print(e)

                    except Exception as e :
                        print(e)
                        a = np.array([math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)])
                        b = np.array([math.floor(float(左目[0])*width), math.floor(float(左目[1])*height)])
                        long = math.floor(float(np.linalg.norm(a-b))*3) #長さ
                        if float(height) >= float(width) :
                            if float(long) >= float(width*0.2) :
                                pass
                            else :
                                try:
                                    a1 = np.array([math.floor(float(左股関節[0])*width), math.floor(float(左股関節[1])*height)])
                                    b1 = np.array([math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)])
                                    long_hi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                    if long >= long_hi_ca :
                                        pass
                                    else :
                                        cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                except Exception as e :
                                    print(e)
                                    try :
                                        a1 = np.array([math.floor(float(右股関節[0])*width), math.floor(float(右股関節[1])*height)])
                                        b1 = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                                        long_mi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                        if long >= long_mi_ca :
                                            pass
                                        else :
                                            cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                    except Exception as e:
                                        print(e)

                        else :
                            if float(long) >= float(height*0.2) :
                                pass
                            else :
                                try:
                                    a1 = np.array([math.floor(float(左股関節[0])*width), math.floor(float(左股関節[1])*height)])
                                    b1 = np.array([math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)])
                                    long_hi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                    if long >= long_hi_ca :
                                        pass
                                    else :
                                        cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                except Exception as e :
                                    print(e)
                                    try :
                                        a1 = np.array([math.floor(float(右股関節[0])*width), math.floor(float(右股関節[1])*height)])
                                        b1 = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                                        long_mi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                        if long >= long_mi_ca :
                                            pass
                                        else :
                                            cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                    except Exception as e:
                                        print(e)

                except Exception as e :
                    print(e)
                    try :
                        a = np.array([math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)])
                        b = np.array([math.floor(float(右目[0])*width), math.floor(float(右目[1])*height)])
                        long = math.floor(float(np.linalg.norm(a-b))*3) #長さ
                        if float(height) >= float(width) :
                            if float(long) >= float(width*0.2) :
                                pass
                            else :
                                try:
                                    a1 = np.array([math.floor(float(左股関節[0])*width), math.floor(float(左股関節[1])*height)])
                                    b1 = np.array([math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)])
                                    long_hi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                    if long >= long_hi_ca :
                                        pass
                                    else :
                                        cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                except Exception as e :
                                    print(e)
                                    try :
                                        a1 = np.array([math.floor(float(右股関節[0])*width), math.floor(float(右股関節[1])*height)])
                                        b1 = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                                        long_mi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                        if long >= long_mi_ca :
                                            pass
                                        else :
                                            cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                    except Exception as e:
                                        print(e)

                        else :
                            if float(long) >= float(height*0.2) :
                                pass
                            else :
                                try:
                                    a1 = np.array([math.floor(float(左股関節[0])*width), math.floor(float(左股関節[1])*height)])
                                    b1 = np.array([math.floor(float(左膝[0])*width), math.floor(float(左膝[1])*height)])
                                    long_hi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                    if long >= long_hi_ca :
                                        pass
                                    else :
                                        cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                except Exception as e :
                                    print(e)
                                    try :
                                        a1 = np.array([math.floor(float(右股関節[0])*width), math.floor(float(右股関節[1])*height)])
                                        b1 = np.array([math.floor(float(右膝[0])*width), math.floor(float(右膝[1])*height)])
                                        long_mi_ca = math.floor(float(np.linalg.norm(a1-b1))*0.5) #長さ
                                        if long >= long_mi_ca :
                                            pass
                                        else :
                                            cv.circle(img, (math.floor(float(頭[0])*width), math.floor(float(頭[1])*height)), long, (100, 33, 2), thickness=-1)
                                    except Exception as e:
                                        print(e)
                    except Exception as e :
                        print(e)


            cv.imwrite(str(pathall) + "/EX_image/pictogram_image" + ".jpg", img)
            print("ピクトグラム保存")
            print("__________________________________________________________")
            print("処理を終了")
            sys.exit()





        except IOError:
            time_11 = time.time()

            predict_time = []

            i11 = 0

            print("少なくとも画像ファイルではないです。")

            if path_video_1 == "exit" :
                sys.exit()
            else :
                pass

            # 動画ファイル読込
            cap = cv.VideoCapture(str(path_video_1))

            if not cap.isOpened():
                # 正常に読み込めたのかチェックする
                # 読み込めたらTrue、失敗ならFalse
                print("動画の読み込み失敗")
                sys.exit()
            n1_1 = 0

            while True :
                remove = input("EX_videoのファイルを削除しますか？[y/n] : ")
                if remove == "y":
                    try :
                        os.remove(str(pathall) + '/EX_video/pictogram.mp4')
                        print("正常に削除できました。")
                        break
                    except :
                        print("ファイルを削除できません。ファイルが存在しないと思われます。")
                        break
                elif remove == "n":
                    break
                else :
                    pass
            img_f = glob.glob(str(pathall) + '/image_first/*.jpg')
            img_s = glob.glob(str(pathall) + '/image_second/*.jpg')
            for i in range(0,len(img_f)) :
                try :
                    os.remove(str(pathall) + '/image_first/' + str(i) + ".jpg")
                except :
                    pass
            for i in range(0,len(img_s)) :
                try :
                    os.remove(str(pathall) + '/image_second/' + str(i) + ".jpg")
                except :
                    pass

            while True:
                # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
                is_image,frame_img = cap.read()
                if is_image:
                    i = str(pathall) + "/image_first/" + str(n1_1) + ".jpg"
                    # 画像を保存
                    cv.imwrite(i , frame_img)
                else:
                    # フレーム画像が読込なかったら終了
                    break
                n1_1 += 1
            cap.release()
            print("動画をフレーム毎に分けれましたぞ")

            # 対象画像読み込み
            img1 = cv.imread(str(pathall) + "/image_first/0.jpg",cv.IMREAD_COLOR)
            # 画像の大きさを取得
            height, width = img1.shape[:2]




            path_file_number=glob.glob(str(pathall) + '/image_first/*.jpg')
            file_count = len(path_file_number)
            print(file_count)

            if file_count % 2 == 1 :#kisu nara
                last_file = str(pathall) + "/image_first/" + str(int(file_count)-1) + ".jpg"
                new_last_file = str(pathall) + "/image_first/" + str(file_count) + ".jpg"
                shutil.copy(last_file, new_last_file)
                half = (int(file_count) + 1) / 2
                file_count += 1
            else :
                half = file_count / 2


            thread1 = threading.Thread(target=proc, args=(0,int(half),n1_1))
            thread2 = threading.Thread(target=proc2, args=(int(half),int(file_count)))
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()







            try :
                cap = cv.VideoCapture(path_video_1)                  # 動画を読み込む
                video_frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
                video_fps = cap.get(cv.CAP_PROP_FPS)                 # フレームレートを取得する
                print(video_fps)
                print("↑fps〜")
                video_sec = video_frame_count / video_fps     # 長さ（秒）を計算する

                command1 = ["python",str(pathall) + "/video.py",str(width),str(height),str(video_fps)]
                proc1 = subprocess.Popen(command1)
                print("呼び出し中")
                proc1.communicate()

                cap.release()

                try :
                    video = VideoFileClip(str(path_video_1))
                    audio = video.audio
                    audio.write_audiofile(str(pathall) + "/ou_vi/ou.mp3")
                except Exception as e :
                    print("音声の抽出に失敗")
                    print(e)

                output = str(pathall) + '/ou_vi/pictogram.mp4'
                try :
                    clip = mp.VideoFileClip(str(output)).subclip()
                    clip.write_videofile(str(pathall) + '/EX_video/pictogram1.mp4', audio=str(pathall) + "/ou_vi/ou.mp3")
                except Exception as e :
                    print("音声との合成に失敗")
                    print(e)
                    try :
                        os.rename(str(output),str(pathall) + '/EX_video/pictogram1.mp4')
                    except Exception as e :
                        print(e)

                try :
                    ffmpeg1 = str(pathall) + '/ffmpeg -i ' + str(pathall) + '/EX_video/pictogram1.mp4 ' + str(pathall) + '/EX_video/pictogram.mp4'
                    #ffmpeg1 = '/Applications/ffmpeg -i ' + str(pathall) + '/EX_video/pictogram1.mp4 ' + str(pathall) + '/EX_video/pictogram.mp4'
                    subprocess.call(str(ffmpeg1), shell=True)

                except Exception as e :
                    print(e)

                for remove_1 in sorted(glob.glob(str(pathall) + "/image_first/" + '/*.jpg')):
                    try :
                        os.remove(remove_1)
                    except Exception as e :
                        print(e)
                print("第一の画像削除")
                for remove_2 in sorted(glob.glob(str(pathall) + "/image_second/" + '/*.jpg')):
                    try :
                        os.remove(remove_2)
                    except Exception as e :
                        print(e)
                print("第二の画像削除")
                try :
                    os.remove(str(pathall) + "/ou_vi/ou.mp3")
                except Exception as e :
                    print(e)
                try :
                    os.remove(str(pathall) + '/ou_vi/pictogram.mp4')
                except Exception as e :
                    print(e)

            except Exception as e :
                print(e)

            try :
                os.remove(str(pathall) + '/EX_video/pictogram1.mp4')
            except Exception as e :
                print(e)



            try :
                time_22 = time.time()
                time_interval2 = time_22 - time_11
                time_interval2 = time.strftime('%d:%H:%M:%S', time.gmtime(math.floor(time_interval2)))
                print(str(time_interval2) + "かかりました")
            except Exception as e:
                print(e)
            print("_______________________________________________________________")
            print("処理を終了")
            print("_______________________________________________________________")
            sys.exit()
