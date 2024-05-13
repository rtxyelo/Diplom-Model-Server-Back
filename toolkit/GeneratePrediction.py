"""Import libraries"""

from roboflow import Roboflow
from ultralytics import YOLO
import torch

import sys
sys.path.insert(0, '../toolkit')

from utils import load_symbol_map, cls_to_symbol, sort_coordinates_by_row, input_spaces, to_rus_symbols

from pyaspeller import YandexSpeller
from autocorrect import Speller

import pyttsx3

"""Model prediction class"""
class Model:
    """Initialize trained model"""
    def __init__(self, trained_model_name='yolov8m5_BEST_V2'):
        self.model_path = f'C:/Users/greyb/OneDrive/Desktop/Potok/Study/Diplom-Model-Server-Back/network/runs/detect/{trained_model_name}/weights/best.pt'
        self.model = YOLO(self.model_path)

    """Predict function"""
    def predict(self, image_path='C:/Users/greyb/OneDrive/Desktop/Potok/Study/Diplom-Model-Server-Back/test/test/prepr_imgs', 
                image_name='prepr_img', 
                image_format='jpg',
                augment=True, 
                save_crop=False, 
                show_conf=False, 
                line_width=1, 
                save_model=True, 
                agnostic_nms=True):
        self.augment = augment
        self.save_crop = save_crop
        self.show_conf = show_conf
        self.bb_line_width = line_width
        self.save_model = save_model
        self.agnostic_nms = agnostic_nms

        self.prediction = self.model.predict(f'{image_path}/{image_name}.{image_format}', 
                                                augment=self.augment, 
                                                save_crop=self.save_crop, 
                                                show_conf=self.show_conf, 
                                                line_width=self.bb_line_width, 
                                                save=self.save_model, 
                                                agnostic_nms=self.agnostic_nms)

        return self.prediction

"""Result from Model.prediction class"""
class MakeResultFromPrediction:
    """Initialize class"""
    def __init__(self, prediction, path_to_symbol_map='C:/Users/greyb/OneDrive/Desktop/Potok/Study/Diplom-Model-Server-Back/data/data.yaml', row_threshold=0.01, space_threshold=0.02):
        self.symbol_map = load_symbol_map(path_to_symbol_map)
        self.prediction = prediction
        self.boxes = self.prediction[0].boxes
        self.row_threshold = row_threshold
        self.space_threshold = space_threshold

    """Initial text predict"""
    def prediction_result(self):
        result_classes = cls_to_symbol(self.boxes.cls, self.symbol_map)
        sorted_result = sort_coordinates_by_row(self.boxes.xywhn, result_classes, self.row_threshold)

        sorted_result_1DIM = []
        for sublist in sorted_result:
            sorted_result_1DIM.extend(sublist)

        sorted_result_1DIM_with_spaces = input_spaces(sorted_result_1DIM, self.space_threshold)
        res_rus_text_list = to_rus_symbols(sorted_result_1DIM_with_spaces)
        result_text_str = ''.join(res_rus_text_list)

        return result_text_str

    """Speller text predict"""
    def text_prediction(self, spell_text=True):
        text = self.prediction_result()

        spell = Speller()
        speller = YandexSpeller()

        if spell_text:
            text = spell(text)
            text = speller.spelled(text)
        
        return text

    """Voice text prediction"""
    def voice_prediction(self, voice_name='C:/Users/greyb/OneDrive/Desktop/Potok/Study/Diplom-Model-Server-Back/app/voice/result_voice', voice_text=False):
        text = self.text_prediction()
        
        engine = pyttsx3.init()
        print("Voice engine init!")
        engine.save_to_file(text , f'{voice_name}.wav')
        print("Voice save!")

        if voice_text:
            engine.say(text)
        
        engine.runAndWait()
        
        return text