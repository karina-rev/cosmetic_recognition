#!/usr/bin/env python3
import pandas as pd
import faiss
import argparse
import utils
import requests
import train_model
import config
import os
import torch
import text_spotting
import pickle5 as pickle
from io import BytesIO
from PIL import Image


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to image to recognize")
ap.add_argument("-f", "--faiss", action='store_true', help="load/reload and train faiss index")
ap.add_argument("-k", "--keywords", action='store_true', help="load/reload products dataset and train keywords")
args = vars(ap.parse_args())

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(image_url):
    """
    Поиск url продукта
    :param image_url: url изобрадения для поиска
    :return: url продукта
    """
    if not os.path.exists(config.PATH_TO_PRODUCT_DATASET):
        train_model.make_products_dataframe()
        train_model.train_products_keywords()

    if not os.path.exists(config.PATH_TO_FAISS_INDEX):
        train_model.train_faiss_index()

    with open(config.PATH_TO_PRODUCT_DATASET, 'rb') as f:
        products = pickle.load(f)

    # ---Поиск с помощью OCR---
    # Получаем слова на изображении и находим количество пересечений со словами каждого продукта из базы
    try:
        http = requests.Session()
        http.mount("https://", utils.adapter)
        response = http.get(image_url)
    except requests.exceptions.ConnectionError:
        print('Connection Error')
        return

    image = Image.open(BytesIO(response.content)).convert('RGB')
    ocr_keywords = text_spotting.search_text(image)
    products['ocr_weight'] = [len(set(ocr_keywords) & set(p)) for p in products['keywords']]

    # Находим процент пересечений для каждого продукта
    # Чем больше процент - тем больше таких же слов у продукта, как и у входного изображения
    # Повышаем общий процент пересечений, умножая на 10.5
    total_sum = products['ocr_weight'].sum()
    products['ocr_weight'] = (products['ocr_weight'] / total_sum) * 10.5

    # ---Поиск с помощью CBIR---
    # Находим расстояния до n_similar (по умолчанию 25) самых похожих изображений из базы и их индексы
    distance, indices = search_img_in_faiss(image)
    distance_sum = sum(distance[0])
    products['faiss_weight'] = 0
    # Считаем процент от общей суммы расстояний
    # Вычитаем из единицы, чтобы больший процент соответствовал более похожему изображению
    for idx in range(0, len(distance[0])):
        products.loc[indices[0][idx], ['faiss_weight']] = 1 - (distance[0][idx] / distance_sum)

    # ---Взвешивание результатов---
    # Вес продукта - среднее между процентом пересечений OCR и процентом расстояния CBIR
    products['result_weight'] = (products['ocr_weight'] + products['faiss_weight']) / 2

    # Считаем среднее для продуктов, которые не были найдены с помощью CBIR,
    # но у которых большой процент пересечения слов OCR
    products.loc[(products['faiss_weight'] == 0.0) & (products['ocr_weight'] >= 0.001), 'result_weight'] = \
        (1.0 + products['ocr_weight']) / 2

    # Возвращаем url продукта, у которого получился самый большой вес
    return products.sort_values(by='result_weight', ascending=False).iloc[0]['url']


def search_img_in_faiss(img, n_similar=25):
    """
    Поиск похожих изображений с помощью индексов faiss
    :param img: изображение для поиска в формате PIL Image
    :param n_similar: количество похожих изображений для вывода; по умолчанию 25
    :return: (distance, indices) для  элементов; остортированы по уменьшению расстрояния
    distance - расстояние от вектора данного изображения до похожих
    indices - индексы похожих изображений
    """
    index = faiss.read_index(config.PATH_TO_FAISS_INDEX)
    input_tensor = utils.transform_torch()(img)
    input_tensor = input_tensor.view(1, *input_tensor.shape)
    with torch.no_grad():
        query_descriptors = utils.pooling_output(input_tensor.to(DEVICE)).cpu().numpy()
        distance, indices = index.search(query_descriptors.reshape(1, 2048), n_similar)
    return distance, indices


if args['faiss']:
    train_model.train_faiss_index()
if args['keywords']:
    train_model.train_products_keywords()
if args['image']:
    print(main(args['image']))
