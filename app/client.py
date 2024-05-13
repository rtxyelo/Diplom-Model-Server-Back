"""Import libraries"""

print('Import libraries...')

import socket

from PIL import Image

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

# Чтение изображения из записанной директории
if os.path.exists(recieve_img):
    image = cv2.imread(recieve_img, cv2.IMREAD_UNCHANGED)
    
    # Предподготовка полученного изображения
    prepr_data.Preprocessing(img_size, blur_value, exposure_values).preprocess_image(image)
    
    # Предсказание модели (сырое)
    prediction = model.predict()

    # Обработанное предсказание модели
    result = gen_predict.MakeResultFromPrediction(prediction).voice_prediction()

    print("Result ", result)