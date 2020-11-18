##!/bin/bash
import pandas as pd
import faiss
import utils
import config
import torch
import time
import logging
import text_spotting
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

index = faiss.read_index(config.PATH_TO_FAISS_INDEX)
products = pd.read_pickle(config.PATH_TO_PRODUCT_DATASET)
text_spotting = text_spotting.TextSpotting().train_network()
logging.basicConfig(filename=config.PATH_TO_LOG_FILE,
                    format='%(asctime)s [%(levelname)s] - %(message)s',
                    level=logging.INFO)


def perform_search(image_bytes):
    """
    Поиск продукта в базе по изображению
    :param image_bytes: изображение для поиска в байтах
    :return: словарь с данными найденного продукта
    """
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    start = time.time()

    executors_list = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        executors_list.append(executor.submit(search_img_in_faiss, image))
        executors_list.append(executor.submit(text_spotting.search_text, image))

    distance, indices = executors_list[0].result()
    ocr_keywords = executors_list[1].result()

    # ---Поиск с помощью OCR---
    # Получаем слова на изображении и находим количество пересечений со словами каждого продукта из базы
    products['ocr_weight'] = [len(set(ocr_keywords) & set(p)) for p in products['keywords']]
    # Находим процент пересечений для каждого продукта
    # Чем больше процент - тем больше таких же слов у продукта, как и у входного изображения
    # Повышаем общий процент пересечений, умножая на 10.5
    total_sum = products['ocr_weight'].sum()
    products['ocr_weight'] = (products['ocr_weight'] / total_sum) * 10.5

    # ---Поиск с помощью CBIR---
    # Находим расстояния до n_similar (по умолчанию 25) самых похожих изображений из базы и их индексы
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
    products.loc[(products['faiss_weight'] == 0.0) &
                 (products['ocr_weight'] >= products[products['ocr_weight'] > 0]['ocr_weight'].mean()),
                 'result_weight'] = (products[products['faiss_weight'] > 0]['faiss_weight'].mean() +
                                     products['ocr_weight']) / 2
    # Возвращаем url продукта, у которого получился самый большой вес
    result_product = products.sort_values(by='result_weight', ascending=False).iloc[0]

    end = time.time()
    logging.info(f'Время поиска продукта: {end - start}')

    return {
        'url': result_product['url'],
        'price': result_product['price'],
        'oldprice': result_product['oldprice'],
        'currencyId': result_product['currencyId'],
        'categoryId': result_product['categoryId'],
        'picture': result_product['picture'],
        'typePrefix': result_product['typePrefix'],
        'model': result_product['model'],
        'vendor': result_product['vendor'],
        'barcode': result_product['barcode'],
        'description': result_product['description']
    }


def search_img_in_faiss(img, n_similar=25):
    """
    Поиск похожих изображений с помощью индексов faiss
    :param img: изображение для поиска в формате PIL Image
    :param n_similar: количество похожих изображений для вывода; по умолчанию 25
    :return: (distance, indices) для  элементов; остортированы по уменьшению расстрояния
    distance - расстояние от вектора данного изображения до похожих
    indices - индексы похожих изображений
    """
    input_tensor = utils.transform_torch()(img)
    input_tensor = input_tensor.view(1, *input_tensor.shape)
    with torch.no_grad():
        query_descriptors = utils.pooling_output(input_tensor.to(DEVICE)).cpu().numpy()
        distance, indices = index.search(query_descriptors.reshape(1, 2048), n_similar)
    return distance, indices
