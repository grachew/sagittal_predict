import streamlit as st

import sys                               # ДЛЯ РАБОТЫ КОДА РАСЧЕТА МОДЕЛИ
sys.stdout.reconfigure(encoding='utf-8') # ДЛЯ РАБОТЫ КОДА РАСЧЕТА МОДЕЛИ - ПРАВИЛЬНАЯ КОДИРОВКА

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
#import matplotlib.image as mpimg



import gdown
import os
#import pylab # модуль для построения графиков
#import time
import glob
#import importlib
import io
#import mpl_toolkits

#from sympy.plotting import plot3d
from PIL import Image # отрисовка изображений
from io import BytesIO #  загрузить модель непосредственно из BytesIO объекта

from PIL import Image  # отрисовка изображений
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sympy import *

                    
from tensorflow.keras.models import Sequential # подключение класса создания модели Sequential 
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model


from tensorflow.keras.layers import Dense,Activation,Dropout,BatchNormalization # основные слои 
from tensorflow.keras.layers import Input

from tensorflow.keras.optimizers import Adam # подключение оптимизатора Адам
from tensorflow.keras import utils # утилиты для to_categorical                               
from tensorflow.keras.preprocessing import image # для отрисовки изображений
#from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model







#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№# ОБЩИЙ ЗАГОЛОВОК

st.subheader("Определение типа патологии сустава в сагиттальной проекции", anchor=None)
st.write("__________________________________________")

#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№# ИСХОДНЫЕ ИЗОБРАЖЕНИЯ

st.markdown('<span style="font-size:24px">Исходные и обработанные изображения</span>', unsafe_allow_html=True)

st.markdown("_Примеры исходных изображений от заказчика_")

image1 = Image.open("0_normal/К104Пс.jpg")
image2 = Image.open("1_sujenie/К100Лс.jpg")
image3 = Image.open("2_raschirenie/К115Лс.jpg")
image4 = Image.open("3_distal/К101Пс.jpg")
image5 = Image.open("4_mesial/К107Пс.jpg")

col1, col2, col3, col4, col5 = st.columns(5)# Создать 5 колонок
# Вывести изображения в соответствующие колонки
with col1:
    st.image(image1, caption='Пример normal', use_column_width=True)
with col2:
    st.image(image2, caption='Пример sujenie', use_column_width=True)
with col3:
    st.image(image1, caption='Пример raschirenie', use_column_width=True)
with col4:
    st.image(image2, caption='Пример distal', use_column_width=True)
with col5:
    st.image(image1, caption='Пример mesial', use_column_width=True)

st.markdown("*количество исходных изображений в папках*")
folders = ["0_normal", "1_sujenie", "2_raschirenie", "3_distal", "4_mesial"]
image_extensions = ['.jpg', '.jpeg', '.png']
image_counts = {}
for folder in folders:
    folder_path = os.path.join(".", folder)  # Полный путь к папке
    image_count = len(glob.glob(os.path.join(folder_path, '*')))  # Количество файлов в папке
    image_counts[folder] = image_count
st.text(image_counts)
st.write("__________________________________________")


#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№# ПРИМЕРЫ ИЗОБРАЖЕНИЙ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ

st.markdown("_Примеры модифицированных изображений, используемых в обучении нейронной сети_")
folder_path = "all_web_images"  # Путь к папке 
images = []  # Список для хранения изображений
jpg_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]  # Получение списка всех файлов JPG в папке "all"
for filename in jpg_files[10:15]:  # Загрузка 5 изображений в список
    file_path = os.path.join(folder_path, filename)
    image = Image.open(file_path)
    images.append(image)
image_container = st.container()  # Создание контейнера для отображения изображений в одну горизонтальную линию
with image_container:  # Отображение изображений в контейнере
    image_placeholders = st.empty()
    # Функция для отображения изображений в одну горизонтальную линию
    def display_images():
        cols = image_placeholders.columns(len(images))
        for i, col in enumerate(cols):
            col.image(images[i], use_column_width=True)
    # Отображение изображений
    display_images()
    # Добавление возможности прокрутки
    st.markdown("""<style> .stHorizontalScrollerElement {overflow-x: auto;} </style>""", unsafe_allow_html=True)
    
def count_photos(folder_path_all):
    photo_extensions = ['.jpg', '.jpeg', '.png', 'gif']
    count = 0
    for file_name in os.listdir(folder_path_all):
        _, extension = os.path.splitext(file_name)
        if extension.lower() in photo_extensions:
            count += 1
    return count
folder_path_al = 'all_web_images'

photo_count = count_photos(folder_path_al)
st.write("Количество изображений к подаче в модель для обучения", photo_count)

#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№# РАЗМЕРНОСТЬ ИЗОБРАЖЕНИЙ

folder_path = 'all_web_images'  # Путь до папки с изображениями
image_dimensions = defaultdict(int)  # Создаем словарь для хранения количества изображений каждой размерности

image_names = os.listdir(folder_path)  # имена всех изображений в папке
for image_name in image_names:  # Проход по каждому изображению
    image_path = os.path.join(folder_path, image_name)  # Формируем полный путь до изображения
    image_size = Image.open(image_path).size  # Получаем размеры изображения
    image_dimensions[image_size] += 1  # Увеличиваем счетчик для данной размерности изображения
for dimension, count in image_dimensions.items():  # Печатаем количество изображений каждой размерности
    st.write("Размерность изображения в пикселях", {dimension}, ",", {count}, "всех изображений")
st.write("__________________________________________")


listic = ["0_normal", "1_sujenie", "2_raschirenie", "3_distal", "4_mesial",
          '5_sujenie__distal', "6_sujenie__mesial", "7_raschirenie__distal", "8_raschirenie__mesial"]
st.markdown("_Полученный в результате работы расширенный список категорий патологий_")
st.write(listic[0],',',listic[1],',',listic[2],',',listic[3],',',listic[4],',',listic[5],',',listic[6],',',listic[7],',',listic[8])
st.write("__________________________________________")

#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№# ЗОГОЛОВОК "ОПРЕДЕЛЕНИЕ ПАТОЛОГИЙ"

st.markdown('<span style="font-size:24px">Поэтапное определение патологии</span>', unsafe_allow_html=True)
st.markdown('<span style="font-size:22px">по снимкам, не принимавшим участие в обучении</span>', unsafe_allow_html=True)




st.markdown("_1 этап_")                                                                                 # 1. ИЗ ОБЛАКА В НОУТБУК
st.markdown("_Загрузка модели нейронной сети из Google облака_")

def make_predictions(model, new_data):
    predictions = model.predict(img_array3)
    st.write("Результаты предсказаний:")
    st.write(predictions)
    arr_x = np.argmax(predictions)
    st.write("предикт по работе модели", listic[arr_x])
    st.write("верный ответ", listic[int(target_file.split("_")[0])])
    st.image(image, caption = file_path)

# Создаем глобальную переменную model
session_state = st.session_state
if 'model' not in session_state:
    session_state.model = None

# Кнопка для загрузки модели
if st.button("Загрузить модель"):
    with st.spinner("Please wait for the download..."):
        FILE_ID = "1UVa7ZXLe9fKwjMW3OetcRjeJu68nrs41"
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", "model_sagittal_qq_web_.h5", quiet=False)
        session_state.model = load_model("model_sagittal_qq_web_.h5")
    st.write("Completed!")
st.write("__________________________________________")
    
    
    
                                                                                                    # 2. СПИСОК И ВЫБОР ПАЦИЕНТА
st.markdown("_2 этап_") 
st.markdown("_Выбор пациента_")
st.write("_Список фамилий пациентов в папке_")

folder_path = "all_q_45"  # укажите путь к папке с изображениями
image_filenames = os.listdir(folder_path)
images_dict1 = {}
for i, image_filename in enumerate(image_filenames):
    name_parts = image_filename.split("_")
    images_dict1[i + 1] = name_parts[-1].split(".")[0]
values_as_str = [str(value) for value in images_dict1.values()]
st.text(values_as_str)

folder_path = "all_q_45"  
image_filenames = os.listdir(folder_path)
images_dict1 = {}
for i, image_filename in enumerate(image_filenames):
    name_parts = image_filename.split("_")
    images_dict1[i + 1] = name_parts[-1].split(".")[0]
values_as_str = sorted((str(value) for value in images_dict1.values()))

#selectbox
option = st.selectbox("_3 Выбрать пациента_", values_as_str)
st.write("Вы выбрали:", option)

# Вывести порядковый номер пациента "имярек" в папке представленных изображений
def get_key(digit, value):
    for count, val in images_dict1.items():
        if val == value:
            return count
key1 = get_key(images_dict1, option)
st.write("Порядковый номер в первичном списке:", key1)
st.write("__________________________________________")



###########################################################################################################################
st.markdown("_3 этап_") 
st.markdown("_Определение патологии по снимкам, не принимавшим участие в обучении_")

folder_path = 'all_q_45'
# Получить список всех файлов в папке
files = os.listdir(folder_path)
# Выбрать файл с порядковым номером 
target_file = files[key1-1]
# Создать полный путь к файлу
file_path = os.path.join(folder_path, target_file)
    # Открыть изображение
image = Image.open(file_path)
img_array = np.array(image).reshape(1,24576) # перевод в массив
    
img_array1 = img_array.astype('float')
img_array2 =img_array1/255 # диапазон значений -> от 0 до 1
img_array3 = img_array2.reshape(1, -1) # Конвертация входных данных в numpy массив

# Кнопка для запуска предсказания
if st.button("Predict"):
    if session_state.model:
        with st.spinner("Please wait for predictions..."):
            make_predictions(session_state.model, img_array3)
    else:
        st.write("Модель не загружена. Пожалуйста, загрузите модель сначала.")
