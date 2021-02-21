import threading
import socketserver
import cv2
import numpy as np
import socket
import pygame
import time
from tensorflow import keras


model = keras.models.load_model('best-cnn-model-0217-7.h5')

class KeyboardControl(socketserver.StreamRequestHandler):
    global img
    def handle(self):
        stream_bytes= b' '
        counts = 0
        try:
            while True:
                start_time = time.time()
                if counts < 3:
                    order = ''
                    order ="|"+order+"|"
                    order= order.encode()
                    self.request.sendall(order)
                    counts += 1
                    print('sending empty data for the first time')

                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')

                if first != -1 and last != -1:

                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    ##print('so far it is working')
                    image_scaled = image.reshape(-1,32,48,1)/255.0
                    print(model.predict(image_scaled))
                    predict = np.argmax(model.predict(image_scaled), axis=-1)
                    print(predict)
                    order = str(predict)
                    order = "|" + order + "|"
                    order = order.encode()
                    self.request.sendall(order)

                    ##print('image received')
                    cv2.imshow('image_1', image)
                    cv2.waitKey(1)
                    last_time = time.time()
                    print(last_time - start_time)
                else:
                    ##print('image corruption occured')
                    pass




                # predict = 'predict'
                # order = str(predict)
                # order ="|"+order+"|"
                # order= order.encode()
                # self.request.sendall(order)


            #
            # key = 'forward'
            # if key == 'forward':
            #     print('forward')
            # elif key == 'backward':
            #     print('backward')
            # elif key == 'left':
            #     print('left')
            # elif key == 'right':
            #     print('right')
            #
            # key = key.encode()
            # self.request.sendall(key)
            #
            #     cv2.imshow('image', image)
            #     cv2.imshow('white', masked_image)
            #     print(round((time.time()-start_time),3))
            #     key = cv2.waitkey(1) & 0xFF
            #     if key == ord('q'):
            #         break
            #     # else:
            #     #     print('else')
            #cv2.destroyAllWindows()

        finally:
            print("Connection closed on thread 1")


class ThreadServer(object):
    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), KeyboardControl)
        server.serve_forever()
    ip=socket.gethostbyname(socket.getfqdn())
    print(ip)
    keyboard_thread = threading.Thread(target=server_thread(ip, 8800))
    keyboard_thread.start()

if __name__ == '__main__':
    ThreadServer()


