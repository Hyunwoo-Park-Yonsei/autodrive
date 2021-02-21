import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras

path = glob.glob("C:/autodrive/0217/*jpg")
images = []
target = []

checking_answers = []
for img in path:
    #img 는 파일이름
    #print(img)

    #파일이름 나눠서 real_name으로 바꿈
    splitted_image = os.path.split(img)
    real_name = splitted_image[-1][:-4]
    #print(real_name)

    #checking_answers 가 번호와 타겟데이터 있는 전체데이터
    #target에는 타겟데이터만
    checking_answers.append(real_name)
    if real_name[-1] == 'W':
        target.append(0)
    elif real_name[-1] == 'A':
        target.append(1)
    elif real_name[-1] == 'D':
        target.append(2)
    elif real_name[-1] == 'S':
        target.append(3)
    else:
        print('error occured')


    blacked_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('img', blacked_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    images.append(blacked_image)

train_input, test_input, train_target, test_target = train_test_split(images, target, random_state=42)


train_input = np.array(train_input)
test_input = np.array(test_input)
train_target = np.array(train_target)
test_target = np.array(test_target)

print(train_input.shape)
train_scaled = train_input.reshape(-1,32,48,1)/ 255.0

test_scaled = test_input.reshape(-1,32,48,1)/ 255.0



print(train_input.shape, test_input.shape, train_target.shape, test_target.shape)



train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled,train_target, test_size=0.1, random_state=42)


model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation= 'relu', padding = 'same', input_shape=(32,48,1)))
model.add(keras.layers.MaxPool2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model-0217-7.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience= 10,restore_best_weights=True)
history = model.fit(train_scaled,train_target, epochs=300, validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb,early_stopping_cb])

# 콜백 최상의 검증 점수를 갖는 모델 저장

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])
plt.legend(['loss','accuracy','val_loss','val_accuracy'])
plt.xlabel('epoch')
#plt.ylabel('loss')
plt.show()

model.save('model_0217-7')
model = keras.models.load_model('best-cnn-model-0217-7.h5')
model.evaluate(val_scaled,val_target)
# 입력된 이미지와 실제 파일의 이미지가 맞는지 확인
# cv2.imshow('img', images[100])
# print(checking_answers[100])
# print(target[100])
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# 데이타 정렬 잘됐나 체크
# def fun(li,i):
#     if li[i] == 0:
#         return 'W'
#     elif li[i] == 1:
#         return 'A'
#     elif li[i] == 2:
#         return 'D'
#     elif li[i] == 3:
#         return 'S'
# for i in range(len(target)):
#     # print(checking_answers[i])
#     # print(fun(target,i))
#     if checking_answers[i][-1] != fun(target,i):
#         print('wrong')



"""
#데이터 다운로드
(train_input,train_target),(test_input,test_target) = keras.datasets.fashion_mnist.load_data()
#데이터 전처리
train_scaled = train_input/255.0
train_scaled = train_scaled.reshape(-1,28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled,train_target, test_size=0.2, random_state=42)

#dense1 이 은닉층이고 100개의 뉴런을 가진 밀집층 활성화 함수를 sigmoid로 지정
# dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
#dense2 가 출력층이고, 10개의 클래스를 분류하기 때문에 10개의 뉴런을 두었다.
# dense2 = keras.layers.Dense(10, activation='softmax')
# dense1 과 2를 sequential 클래스에 추가하여 심층신경망 Deep Neural network를 만들었다.
#model = keras.Sequential([dense1,dense2])

# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28,28)))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))
#
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= 'accuracy')
# model.fit(train_scaled, train_target, epochs=5)
# print(model.evaluate(val_scaled, val_target))

# model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
# model.fit(train_scaled, train_target, epochs=5)
# model.evaluate(val_scaled,val_target)

def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(100, activation= 'relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

model = model_fn(keras.layers.Dropout(0.3))
model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=10, verbose=1, validation_data=(val_scaled,val_target))

model.save_weights('model-weights.h5')
model.save('model-whole.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel(' loss, val_loss')
plt.legend(['train','val'])
plt.show()
"""