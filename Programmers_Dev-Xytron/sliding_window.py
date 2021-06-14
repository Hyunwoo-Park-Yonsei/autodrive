#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, random, math, copy

Width = 640
Height = 480

#동영상 읽기
cap = cv2.VideoCature("xycar_track1.mp4")
window_title = 'camera'

#와핑한 이미지의 사이즈 bird eye view로 본 것의 이미지의 사이즈
warp_img_w = 320
warp_img_h = 240

#와핑할때 의 margin 값
warpx_margin =20
warpy_margin =3

#슬라이디 윈도우 개수, 슬라이딩 윈도우의 넓이, 선을 그릴 때의 threshold 값
nwindows = 9
margin =12
minpix = 5

lane_bin_th = 145

# bird eye view로 변환 작업

# 와핑할 영역 선정
warp_src = np.array([
    [230-warpx_margin, 300-warpy_margin],
    [45-warpx_margin, 450+warpy_margin],
    [445-warpx_margin, 300+warpy_margin],
    [610-warpx_margin, 450+warpy_margin],
], dtype=np.float32)
# 결과 이미지 크기 선정
warp_dist = np.array([
    [0,0],
    [0,warp_img_h],
    [warp_img_w,0],
    [warp_img_w,warp_img_h],
],dtype=np.float32)

calibrated =True
#자이카 카메라 왜곡에 의한 calibration
if calibrated:
    mtx = np.array([
        [422.037858,0.0,245.895397],
        [0.0,435.589734,163.625535],
        [0.0,0.0,1.0]
    ])
    dist = np.array([-0.289296,0.061035,0.001786,0.15238,0.0])

    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width,Height),1,(Width,Height))

#왜곡된 이미지를 펴는 함수
def calibrate_image(frame):
    global Width, Height
    global mtx, dist
    global cal_mtx, cal_roi

    tf_image = cv2.undistort(frame,mtx,dist,None,cal_mtx)
    x,y,w,h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]

    return cv2.resize(tf_image,(Wdith,Height))

#변환 전후의 4개점 좌표를 전달해 새로운 이미지로 만든다
def warp_image(img,src,dst,size):
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warp_img = cv2.warpPerspective(img,M,size,flags=cv2.INTER_LINEAR)

    return warp_img,M,Minv

def warp_process_image(img):
    global nwindows
    global margin
    global minpix
    global lane_bin_th
    #가우시안 블러로 노이즈 제거
    blur = cv2.GaussianBlur(img,(5,5),0)

    #HLS포맷에서 흰색선 구분 쉬워서 L채널을 사용
    _,L,_ = cv2.split(cv2.cvtColor(blur,cv2.COLOR_BGR2HLS))
    #L채널을 확실하게 하기 위해 이진화한다.
    _, lane = cv2.thershold(L,lane_bin_th, 255, cv2.THRESH_BINARY)

    #추출된 이미지를 히스토그램화한다
    histogram = np.sum(lane[lane.shape[0]//2:,:], axis = 0)
    #x좌표를 반으로 나누어 왼쪽차선과 오른쪽차선 구분한다
    midpoint = np.int(histogram.shape[0]/2)
    #왼쪽차선중 흰색 픽셀이 가장 많은 지점을 왼쪽 시작지점으로 잡는다
    leftx_current = np.argmax(histogram[:midpoint])
    #오른쪽차선중 흰색 픽셀이 가장 많은 지점을 오른쪽 시작지점으로 잡는다
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint


    #차선의 위치에 슬라이딩 윈도우를 그린다

    #윈도우 하나의 크기 설정
    window_height = np.int(lane.shape[0]/nwindows)
    nz = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []

    lx,ly,rx,ry = [], [], [], []
    out_img = np.dstack((lane,lane,lane))*255
    #윈도우 그리기기
   for window in range(nwindows):
        win_yl = lane.shape[0] - (window+1)*window_height
        win_yh = lane.shape[0] - (window) * window_height

        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin

        cv2.rectangle(out_img, (win_xll,win_yl),(win_xlh,win_yh),(0,255,0),2)
        cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)

        #픽셀의 x 좌표를 모은다
        good_left_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) &
                          (nz[1] >= win_xll) & (nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) &
                          (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        #흰색점이 5개 이상인 경우일때 x좌표의 평균값을 구한다.

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nz[1][good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nz[1][good_right_inds]))

        lx.append(leftx_current)
        ly.append((win_yl + win_yh)/2)
        rx.append(rightx_current)
        ry.append((win_yl + win_yh)/2)

    # 모은 점의 좌표를 통해 2차함수를 fit한다
   left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    lfit = np.polyfit(np.array(ly), np.array(lx),2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)

# 구한 lfit rfit을 다시 원근 변환하여 원래 이미지에 덧그린다
def draw_line(image,warp_img,MInv,left_fit,right_fit):
    global Width, Height
    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax -1,yMax)
    color_warp = np.zeros_like(warp_img).astype(np,.uint8)

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    pts = np.hstack((pts_left,pts_right))

    color_warp = cv2.fillPoly(color_warp,np.int_([pts]), (0,255,0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))

    return cv2.addWeighted(image,1,newwarp, 0.3,0)

def start():
    global Width, Height, cap

    _, frame = cap.read()
    while not frame.size == (Width*Height*3):
        _, frame = cap.read()
        continue
    while cap.isOpened():
        _, frame = cap.read()

        image = calibrate_image(frame)
        warp_img, M, Minv = warp_image(image,warp_src,warp_dist,(warp_img_w,warp_img_h))

        left_fit, right_fit = warp_process_image(warp_img)
        lane_img = draw_line(iamge,warp_img,Minv,left_fit,right_fit)

        cv2.imshow(window_title, lane_img)
        cv2.waitkey(1)
if __name__ == '__main__':
    start()
























