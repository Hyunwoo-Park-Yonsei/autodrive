import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
import pygame
import struct
import cv2
import io
from picamera import PiCamera
import picamera
from socket import *
import argparse
import socket
import sys,traceback
import logging
logging.basicConfig(level=logging.ERROR)


parser = argparse.ArgumentParser(description='Press Ip aress and Port number')
parser.add_argument('-ip', type=str, default = "192.168.35.245")
parser.add_argument('-port', type=int, default = 8800)
a = parser.parse_args()
ip = a.ip
port = a.port


# motor1A = 16
# motor1B = 18
# motor2A = 15
# motor2B = 13
# GPIO_TRIGGER = 10
# GPIO_ECHO = 12
# 
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(motor1A,GPIO.OUT)
# GPIO.setup(motor1B,GPIO.OUT)
# GPIO.setup(motor2A,GPIO.OUT)
# GPIO.setup(motor2B,GPIO.OUT)
# 
# p1A = GPIO.PWM(motor1A, 200)
# p1B = GPIO.PWM(motor1B, 200)
# p2A = GPIO.PWM(motor2A, 200)
# p2B = GPIO.PWM(motor2B, 200)
# GPIO.setup(GPIO_TRIGGER,GPIO.OUT)
# GPIO.setup(GPIO_ECHO,GPIO.IN)
# GPIO.output(GPIO_TRIGGER,False)
# p1A.start(0)
# p1B.start(0)
# p2A.start(0)
# p2B.start(0)
# 
# left=0
# right=0

# 모터 상태
STOP  = 0
FORWARD  = 1
BACKWORD = 2
LEFT =3
RIGHT =4

# 모터 채널
CH1 = 0
CH2 = 1

# PIN 입출력 설정
OUTPUT = 1
INPUT = 0

# PIN 설정
HIGH = 1
LOW = 0

# 실제 핀 정의
#PWM PIN
ENA = 26  #37 pin
ENB = 0   #27 pin

#GPIO PIN
IN1 = 19  #37 pin
IN2 = 13  #35 pin
IN3 = 5   #31 pin
IN4 = 6   #29 pin

# 핀 설정 함수
def setPinConfig(EN, INA, INB):        
    GPIO.setup(EN, GPIO.OUT)
    GPIO.setup(INA, GPIO.OUT)
    GPIO.setup(INB, GPIO.OUT)
    # 100khz 로 PWM 동작 시킴 
    pwm = GPIO.PWM(EN, 100) 
    # 우선 PWM 멈춤.   
    pwm.start(0) 
    return pwm

# 모터 제어 함수
def setMotorContorl(pwm, INA, INB, speed, stat):

    #모터 속도 제어 PWM
    pwm.ChangeDutyCycle(speed)  
    
    if stat == FORWARD:
        GPIO.output(INA, HIGH)
        GPIO.output(INB, LOW)
    
    if stat == LEFT:
        GPIO.output(INA, HIGH)
        GPIO.output(INB, LOW)
        
    if stat == RIGHT:
        GPIO.output(INA, HIGH)
        GPIO.output(INB, LOW)
    #뒤로
    elif stat == BACKWORD:
        GPIO.output(INA, LOW)
        GPIO.output(INB, HIGH)
        
    #정지
    elif stat == STOP:
        GPIO.output(INA, LOW)
        GPIO.output(INB, LOW)

        
# 모터 제어함수 간단하게 사용하기 위해 한번더 래핑(감쌈)
def setMotor(ch, speed, stat):
    if ch == CH1 and stat == RIGHT:
        #pwmA는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
        setMotorContorl(pwmA, IN2, IN1, speed, stat)
    elif ch == CH1:
        setMotorContorl(pwmA, IN1, IN2, speed, stat)
    
    elif ch == CH2 and stat == LEFT:
        setMotorContorl(pwmB, IN4, IN3, speed, stat)
    

    elif ch == CH2:
        #pwmB는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
        setMotorContorl(pwmB, IN3, IN4, speed, stat)
  

# GPIO 모드 설정 
GPIO.setmode(GPIO.BCM)
      
#모터 핀 설정
#핀 설정후 PWM 핸들 얻어옴 
pwmA = setPinConfig(ENA, IN1, IN2)
pwmB = setPinConfig(ENB, IN3, IN4)


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ip, port))
connection = client_socket.makefile('wb')

time.sleep(.1)

try:
    with picamera.PiCamera() as camera:
        camera.resolution = (32,48)
        camera.vflip = False
        camera.hflip = False
        camera.framerate = 30
        camera.start_preview()
        time.sleep(2)
        
        stream = io.BytesIO()
        for foo in camera.capture_continuous(stream, 'jpeg',use_video_port = True):
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            stream.seek(0)
            connection.write(stream.read())
            stream.seek(0)
            stream.truncate()
#             connection.write(struct.pack('<L',0))
            
#             data = client_socket.recv(1024)
            
            try:
                data = client_socket.recv(1024)
                data = str(data)
                data = data.split('|')[1]
                print(data)
                predict = int(data[1])
                print(predict)
            
#                 if predict == 1:
#                     setMotor(CH1, 60, LEFT)
#                     setMotor(CH2, 60, LEFT)
#                     stat = 'A'
#                     print('left')
# 
#                 elif predict == 2:
#                     setMotor(CH1, 60, RIGHT)
#                     setMotor(CH2, 60, RIGHT)
#                     stat = 'D'
#                     print('right')
# 
#                 elif predict == 0:
#                     setMotor(CH1, 60, FORWARD)
#                     setMotor(CH2, 60, FORWARD)
#                     stat = 'W'
#                     print('forward')
# 
#                 elif predict == 3:
#                     setMotor(CH1, 60, BACKWORD)
#                     setMotor(CH2, 60, BACKWORD)
#                     stat = 'S'
#                     print('backward')
#                 else:
#                     setMotor(CH1, 0, STOP)
#                     setMotor(CH2, 0, STOP)
#                     stat = 'X'
            except:
                print('exceptt')
                pass

#         except:
#             logging.error(traceback.format_exc())
#             traceback.print_exc()
#             print('except')
                          
        
    connection.write(struct.pack('<L',0))
                    
except:
    logging.error(traceback.format_exc())
    print('excepttt')
finally:
    client_socket.close()
    connection.close()

GPIO.cleanup()



