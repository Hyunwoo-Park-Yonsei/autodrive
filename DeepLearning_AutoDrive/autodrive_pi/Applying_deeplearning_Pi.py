import sys
# 라즈베리파이 GPIO 패키지 
import RPi.GPIO as GPIO
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2
import numpy as np
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
from tensorflow import keras



camera = PiCamera()
camera.resolution = (48,32)
camera.vflip = False
camera.hflip = False
camera.framerate =30

rawCapture = PiRGBArray(camera, size = (48,32))






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



#제어 시작
model = keras.models.load_model('best-cnn-model-11.h5')



stat = 'X'

for frame in camera.capture_continuous(rawCapture, format= "bgr", use_video_port=True):
    #time_1 = time.time()
    image = frame.array
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.array(gray_image)
#     cv2.imshow('gray_img',gray_image
    gray_image_scaled = gray_image.reshape(-1,32,48,1)/255.0
    
    predict = np.argmax(model.predict(gray_image_scaled), axis=-1) 
    #print(model.predict(gray_image_scaled))
    
    key =cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    
    if key == ord('q'):
        break
    
    if predict == 1:
        setMotor(CH1, 60, LEFT)
        setMotor(CH2, 60, LEFT)
        stat = 'A'
        #print('left')

    elif predict == 2:
        setMotor(CH1, 60, RIGHT)
        setMotor(CH2, 60, RIGHT)
        stat = 'D'
        #print('right')

    elif predict == 0:
        setMotor(CH1, 60, FORWARD)
        setMotor(CH2, 60, FORWARD)
        stat = 'W'
        #print('forward')

    elif predict == 3:
        setMotor(CH1, 60, BACKWORD)
        setMotor(CH2, 60, BACKWORD)
        stat = 'S'
        #print('backward')
    else:
        setMotor(CH1, 0, STOP)
        setMotor(CH2, 0, STOP)
        stat = 'X'
#     setMotor(CH1, 0, STOP)
#     setMotor(CH2, 0, STOP)
#     time.sleep(1)
        #print('stop')
#     time_2 = time.time()
#     print(time_2-time_1)



# 종료
GPIO.cleanup()


