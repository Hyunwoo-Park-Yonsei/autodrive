#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, random, math, time

Width = 640
Height = 480
Offset = 340


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로채움 
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img, lines


# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    cv2.rectangle(img, (lpos - 5, -5 + offset),
                  (lpos + 5, +5 + offset),
                  (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, -5 + offset),
                  (rpos + 5, +5 + offset),
                  (0, 255, 0), 2)

    return img


def draw_rectangle2(img, lpos, rpos, offset=0):
    cv2.rectangle(img, (lpos - 5, -5 + offset),
                  (lpos + 5, +5 + offset),
                  (255, 255, 255), 2)
    cv2.rectangle(img, (rpos - 5, -5 + offset),
                  (rpos + 5, +5 + offset),
                  (255, 255, 255), 2)

    return img


def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def lane_detect(frame):
    global Offset
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_img = gaussian_blur(gray, 3)
    canny_img = canny(blur_img, 100, 240)

    vertices = np.array(
        [[(5, Offset + 10), (50, Height / 2 + 50), (Width - 50, Height / 2 + 50), (Width - 10, Offset + 10), ]],
        dtype=np.int32)

    ROI_img = region_of_interest(canny_img, vertices)
    cv2.imshow("only canny", canny_img)

    hough_img, line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 2, 50)
    # cv2.imshow("hough",hough_img)
    line_arr = np.squeeze(line_arr)
    try:
        slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
        avg_angle = sum(slope_degree) / len(slope_degree)
    except:
        avg_angle = 0
    offset_line = ROI_img[Offset, :]
    reverse_line = offset_line[::-1]
    left_line = offset_line
    right_line = reverse_line

    rpos = 639 - np.argmax(right_line)
    lpos = np.argmax(left_line)

    hough_img[Offset, :] = 255
    hough_img[Height - 100, 0] = 255
    hough_img[Height / 2, Width / 2 - 50,] = 255
    hough_img[Height / 2, Width / 2 + 100] = 255
    hough_img[Height - 130, Width - 15] = 255

    if rpos < 448 and lpos > 153:
        if Width - rpos > lpos:
            print('a')
            print(lpos, rpos)
            rpos = -1
        else:
            print('b')
            print(lpos, rpos)
            lpos = -1
    elif avg_angle > 50 or rpos < 390:
        rpos = -1
        print('right', avg_angle)
    elif avg_angle < -50 or lpos > 152:
        lpos = -1
        print('left', avg_angle)
    else:
        print(avg_angle)

    if lpos == -1:
        lpos = rpos - 475
    elif rpos == -1:
        rpos = lpos + 475
    final = draw_rectangle(hough_img, lpos, rpos, Offset)

    # cv2.imshow("final", final)
    # print(slope_degree)
    print(lpos, rpos)

    ROI_img = draw_rectangle2(ROI_img, lpos, rpos, Offset)
    cv2.imshow("canny,roi", ROI_img)
    return lpos, rpos


# You are to find "left and light position" of road lanes
def process_image(frame):
    global Offset

    # lpos, rpos = 100,500
    lpos, rpos = lane_detect(frame)
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)
    # cv2.imshow('frame', frame)

    return (lpos, rpos), frame


def draw_steer(image, steer_angle):
    global Width, Height, arrow_pic

    arrow_pic = cv2.imread('steer_arrow.png', cv2.IMREAD_COLOR)

    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height / 2
    arrow_Width = (arrow_Height * 462) / 728

    matrix = cv2.getRotationMatrix2D((origin_Width / 2, steer_wheel_center), (steer_angle) * 1.5, 0.7)
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width + 60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = image[arrow_Height: Height, (Width / 2 - arrow_Width / 2): (Width / 2 + arrow_Width / 2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow_pic)
    image[(Height - arrow_Height): Height, (Width / 2 - arrow_Width / 2): (Width / 2 + arrow_Width / 2)] = res

    cv2.imshow('steer', image)


# You are to publish "steer_anlge" following load lanes
if __name__ == '__main__':
    cap = cv2.VideoCapture('kmu_track.mkv')
    time.sleep(3)

    while not rospy.is_shutdown():
        ret, image = cap.read()
        pos, frame = process_image(image)
        # CCW +
        steer_angle = 5 - (sum(pos) - 660) / 5
        draw_steer(frame, steer_angle)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break