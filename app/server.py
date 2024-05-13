"""Import libraries"""

print('Import libraries...')

import socket

import base64

import os
import cv2

import sys
sys.path.insert(0, '../toolkit')

import PreprocessData as prepr_data
import GeneratePrediction as gen_predict

print('Libraries are imported...')
                
              
"""Import pretarined model"""
model = gen_predict.Model()


"""Default config"""
img_size = 1024
blur_value = 0
exposure_values = [0.45, 50]

row_threshold=0.01, 
space_threshold=0.02

recieve_img = 'received_image.jpg'


"""Preprocess image class"""
preprocess = prepr_data.Preprocessing(img_size, blur_value, exposure_values)


print('Readry to predict...')


def send_audio(conn, filename='C:/Users/greyb/OneDrive/Desktop/Potok/Study/Diplom-Model-Server-Back/app/voice/result_voice.wav'):
    with open(filename, 'rb') as f:
        # Определение размера файла
        file_size = os.path.getsize(filename)
        print("Voice size ", file_size)
        
        # Отправка размера файла
        audio_size_bytes = file_size.to_bytes(4, byteorder='little')
        conn.sendall(audio_size_bytes)

        good_answer = 0
        while good_answer == 0:
            good_answer_bytes = conn.recv(4)  # Предполагается, что размер передается как 4 байта (int32)
            good_answer = int.from_bytes(good_answer_bytes, byteorder='little')
            print("Answer ", good_answer)
            if good_answer == 1:
                break
            else:
                conn.sendall(audio_size_bytes)
                print("Voice size in bytes ", audio_size_bytes)


        # Отправка размера файла
        #audio_size_bytes = file_size.to_bytes(4, byteorder='little')
        #conn.sendall(audio_size_bytes)
        #print("Voice size in bytes ", audio_size_bytes)

        # Отправка аудио данных
        send_size = 0
        while send_size < file_size:
            data = f.read(1024)
            #print("Voice data ", len(data))
            if not data:
                break
            conn.sendall(data)
            send_size += len(data)


def handle_client_connection(conn, addr):
    try:

        # Получение размера изображения
        size_bytes = conn.recv(4)  # Предполагается, что размер передается как 4 байта (int32)
        image_size = int.from_bytes(size_bytes, byteorder='little')  # Преобразование байт в число
        print("image_size ", image_size)

        # Получение самого изображения
        received_size = 0
        with open(recieve_img, 'wb') as file:
            while received_size < image_size:
                data = conn.recv(1024)
                if not data:
                    break
                file.write(data)
                received_size += len(data)
            print("received_size ", received_size)

        # Чтение изображения из записанной директории
        if os.path.exists(recieve_img):
            image = cv2.imread(recieve_img, cv2.IMREAD_UNCHANGED)

            # Предподготовка полученного изображения
            prepr_data.Preprocessing(img_size, blur_value, exposure_values).preprocess_image(image)

            # Предсказание модели (сырое)
            prediction = model.predict()

            # Обработанное предсказание модели
            result_text = gen_predict.MakeResultFromPrediction(prediction).voice_prediction()

            print("Result ", len(result_text), " ", result_text)

            # Отправка текста клиенту
            conn.sendall(result_text.encode())
            print('Text sent successfully!')

            # Отправка звука клиенту
            send_audio(conn)
            print('Sound sent successfully!')
            
        else:
            print('Image file not found!')
            
    except Exception as e:
        print('Error with client:', e)


def open_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    hostname = socket.gethostname()
    port = 12345
    server.bind((hostname, port))
    server.listen(2)
    print('Server is working...')

    conn, addr = server.accept()
    print('Client is connected...')

    return conn, addr


def close_server(server):
    server.close()
    print("Server closed.")



def start_server():
    try:
        while True:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            hostname = socket.gethostname()
            port = 12345
            server.bind((hostname, port))
            server.listen(1)
            print('Server is working...')

            conn, addr = server.accept()
            print('Client is connected...')

            #conn, addr, server = open_server()
            handle_client_connection(conn, addr)
            #close_server(server)

            server.close()
            print("Server closed.")

    except Exception as e:
        print('Error with server:', e)



if __name__ == '__main__':
    start_server()