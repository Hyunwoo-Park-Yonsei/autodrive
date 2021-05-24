#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, random, math, time
import collections

pos_tmp = collections.deque([])

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
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)
    return img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    return lines  # , line_img


# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    cv2.rectangle(img, (lpos - 5, -5 + offset),
                  (lpos + 5, +5 + offset),
                  (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, -5 + offset),
                  (rpos + 5, +5 + offset),
                  (0, 255, 0), 2)

    return img


def draw_rectangle2(img, pos, offset=0):
    cv2.rectangle(img, (pos - 5, -5 + offset),
                  (pos + 5, +5 + offset),
                  (255, 0, 0), 2)
    return img


def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def weighted_img(img, initial_img, a=1, b=1, c=0):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, a, img, b, c)


def lane_detect(frame):
    global Offset
    global pos_tmp

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_img = gaussian_blur(gray, 3)
    canny_img = canny(blur_img, 50, 240)
    # cv2.imshow("canny",canny_img)

    vertices = np.array(
        [[(35, Offset + 10), (80, Height / 2 + 50), (Width - 80, Height / 2 + 50), (Width - 10, Offset + 10), ]],
        dtype=np.int32)

    ROI_img = region_of_interest(canny_img, vertices)
    # cv2.imshow("only canny",canny_img)
    line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 2, 30)
    # cv2.imshow("hough",hough_img)
    line_arr = np.squeeze(line_arr)
    try:
        slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
        avg_angle = sum(slope_degree) / len(slope_degree)
    except:
        avg_angle = 0
        slope_degree = [90 for _ in range(10)]

        ##############
    left = []
    right = []
    line_arr = line_arr.tolist()
    main_line = []
    try:
        for i in range(len(line_arr)):
            overlapped = False
            if abs(abs(slope_degree[i]) - 90) > 20 and abs(abs(slope_degree[i]) - 180) > 10:
                if len(main_line) == 0:
                    main_line.append(line_arr[i])
                for j in main_line:
                    area = abs((j[0] - line_arr[i][0]) * (j[3] - line_arr[i][1]) - (j[1] - line_arr[i][1]) * (
                                j[2] - line_arr[i][0]))
                    length = ((j[0] - j[2]) ** 2 + (j[1] - j[3]) ** 2) ** 0.5
                    dist = area / length
                    if dist < 10:
                        overlapped = True
                if not overlapped:
                    main_line.append(line_arr[i])
        if len(main_line) == 3:
            main_line = sorted(main_line, key=lambda x: x[0])
            if main_line[2][0] - main_line[1][0] < 230:
                # print('this is popped',main_line.pop())
                main_line.pop()
            elif 200 < main_line[1][0] < 250:
                # print('this is popped',main_line.pop(1))
                main_line.pop(1)
    except:
        pass

    ############

    line_img = np.zeros((ROI_img.shape[0], ROI_img.shape[1], 3), dtype=np.uint8)
    try:
        ffinal = draw_lines(line_img, main_line)
    except:
        print("no line!")
        ffinal = line_img

    offset_line = ROI_img[Offset, :]
    reverse_line = offset_line[::-1]
    left_line = offset_line
    right_line = reverse_line
    rpos = 639 - np.argmax(right_line)
    lpos = np.argmax(left_line)
    #    ffinal[Offset, :] = 255
    #    ffinal[Offset + 10, 35] = 255
    #    ffinal[Height / 2 + 50, 100] = 255
    #    ffinal[Height / 2 + 50, Width - 80] = 255
    #    ffinal[Offset + 10, Width - 10] = 255

    if rpos < 448 and lpos > 153:
        if Width - rpos > lpos:
            rpos = -1
        else:
            lpos = -1
    elif avg_angle > 50 or rpos < 390:
        rpos = -1
    elif avg_angle < -50 or lpos > 152:
        lpos = -1

    if lpos == -1:
        lpos = rpos - 475
    elif rpos == -1:
        rpos = lpos + 475

    if pos_tmp:
        gradient = lpos - pos_tmp[0][0], rpos - pos_tmp[0][1]
        if abs(gradient[0]) < 29 and abs(gradient[1]) < 29:
            pos_tmp.pop()
            pos_tmp.append([lpos, rpos])
        elif abs(gradient[0]) > 29 and abs(gradient[1]) < 29:
            lpos = pos_tmp[0][0]

        elif abs(gradient[0]) < 29 and abs(gradient[1]) > 29:
            rpos = pos_tmp[0][1]
        else:
            lpos = pos_tmp[0][0]
            rpos = pos_tmp[0][1]

    else:
        print('else')
        pos_tmp.append([lpos, rpos])

    fffinal = draw_rectangle(ffinal, lpos, rpos, Offset)
    pos = (lpos + rpos) // 2
    final = draw_rectangle2(fffinal, pos, Offset)
    # cv2.imshow('ff',ffinal)
    # cv2.imshow('roi',ROI_img)

    #    for main in main_line:
    #        print(main)
    #    for i in range(len(line_arr)):
    #        print(line_arr[i],slope_degree[i])

    return lpos, rpos, final, len(main_line)


# You are to find "left and light position" of road lanes
def process_image(frame):
    global Offset

    # lpos, rpos = 100,500
    lpos, rpos, final, line_num = lane_detect(frame)
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)
    # cv2.imshow('frame', frame)

    return [lpos, rpos], frame, final, line_num


def draw_steer(image, steer_angle, final):
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
    result = weighted_img(final, image)
    cv2.imshow('steer', result)


# You are to publish "steer_angle" following load lanes
if __name__ == '__main__':
    cap = cv2.VideoCapture('kmu_track.mkv')
    time.sleep(3)
    deque = collections.deque([])
    while not rospy.is_shutdown():
        ret, image = cap.read()
        pos, frame, final, line_num = process_image(image)
        # CCW +

        steer_angle = 5 - (sum(pos) - 660) / 5
        if steer_angle > 50:
            steer_angle = 50
        elif steer_angle < -50:
            steer_angel = -50
        #        if len(deque) > 1:
        #            tmp = deque.popleft()
        #            deque.append(steer_angle)
        #            gradient = deque[1]-deque[0]
        #            if abs(gradient) > 7:
        #                deque.appendleft(tmp)
        #                deque.pop()
        #        else:
        #            deque.append(steer_angle)
        #        new_steer_angle = deque[-1]
        #        if abs(new_steer_angle) > 50:
        #            if new_steer_angle > 0:
        #                new_steer_angle = 50
        #            else:
        #                new_steer_angle = -50
        # print(new_steer_angle,deque,line_num)
        #       if line_num>4:
        #            new_steer_angle = 0
        #            deque.pop()
        #            deque.append(0)
        draw_steer(frame, steer_angle, final)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
