import pandas as pd
import pickle
import keywords_utils
import argparse
import glob
import train_model
import config
from _collections import defaultdict
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to image to recognize")
ap.add_argument("-c", "--cbir", action='store_true', help="load/reload CBIR model")
ap.add_argument("-k", "--keywords", action='store_true', help="load/reload products keywords dataset")
args = vars(ap.parse_args())


# загрузка CBIR модели
if not args['keywords'] and not args['cbir']:
    try:
        cbir = pickle.load(open(config.PATH_TO_CBIR_MODEL, 'rb'))
    except FileNotFoundError:
        train_model.train_cbir_model()
        cbir = pickle.load(open(config.PATH_TO_CBIR_MODEL, 'rb'))

# загрузка датасета products с найденными словами
if not args['keywords'] and not args['cbir']:
    try:
        products = pd.read_pickle(config.PATH_TO_PRODUCT_DATASET)
    except FileNotFoundError:
        train_model.generate_products_keywords()
        products = pd.read_pickle(config.PATH_TO_PRODUCT_DATASET)


def add_new_products():
    """
    Добавление новых продуктов в базу
    Создаются новые файлы output/cbir.pkl, output/product_keywords.pkl
    """
    global cbir, products
    files_in_directory = [x.split('/', 1)[1] for x in glob.glob(config.PATH_TO_FILES + '/*/*')]
    files_in_dataset = products['path'].to_list()
    difference = set(files_in_directory) - set(files_in_dataset)

    if difference:
        train_model.train_cbir_model()
        train_model.add_products_keywords(difference)
        cbir = pickle.load(open(config.PATH_TO_CBIR_MODEL, 'rb'))
        products = pd.read_pickle(config.PATH_TO_PRODUCT_DATASET)


def perform_search(image):
    """
    Поиск названия продукта
    :param image: изображение для поиска
    :return: название продукта
    """
    global cbir, products
    try:
        image = Image.open(image)
    except FileNotFoundError:
        return 'По указанному пути изображения не существует'

    # ---Поиск с помощью OCR---
    # Получаем слова на изображении и находим количество пересечений со словами каждого продукта из базы
    ocr_keywords = keywords_utils.get_keywords(image)
    products['intersection'] = [len(ocr_keywords & p) for p in products['keywords']]
    # Находим процент пересечений для каждого продукта
    # Чем больше процент - тем больше таких же слов у продукта, как и у входного изображения
    # Повышаем общий процент пересечений, умножая на 1.5
    total_sum = products.groupby(by='name')['intersection'].sum().sum()
    ocr_weighted_products = (products.groupby(by='name')['intersection'].sum().sort_values(ascending=False)
                             / total_sum) * 1.5
    ocr_weighted_products.fillna(0, inplace=True)

    # ---Поиск с помощью CBIR---
    # Находим расстояния до 9 самых похожих изображений из базы и их индексы
    distance, indices = cbir.search_img(image)
    distance_sum = sum(distance[0])
    cbir_weighted_names = defaultdict(int)
    # Считаем процент от общей суммы расстояний
    # Вычитаем из единицы, чтобы больший процент соответствовал более похожему изображению
    for idx in range(0, 9):
        name = cbir.image_paths[indices[0][idx]][0].split('/')[1]
        weight = 1 - (distance[0][idx] / distance_sum)
        cbir_weighted_names[name] = cbir_weighted_names[name] + weight

    # ---Взвешивание результатов---
    overall_weighted = {}
    # Вес продукта - среднее между процентом пересечений OCR и процентом расстояния CBIR
    # Берем во внимание проценты пересечений больше 0.15 (нижний порог), чтобы отсечь случайные пересечения
    # Сначала считаем среднее для 9 самых похожих изображений, найденных с помощью CBIR
    for key, value in cbir_weighted_names.items():
        if ocr_weighted_products[key] > 0.15:
            ocr_weight = ocr_weighted_products[key]
        else:
            ocr_weight = 0.0
        overall_weighted[key] = (cbir_weighted_names[key] + ocr_weight) / 2

    # Считаем среднее для продуктов, которые не были найдены с помощью CBIR,
    # но у которых большой процент пересечения слов OCR
    for key, value in ocr_weighted_products[ocr_weighted_products > 0.15].items():
        if key not in overall_weighted:
            overall_weighted[key] = (1.0 + value) / 2

    # Возвращаем название продукта, у которого получился самый большой вес
    return sorted(overall_weighted.items(), key=lambda item: item[1], reverse=True)[0][0]


if args['cbir']:
    train_model.train_cbir_model()
    cbir = pickle.load(open(config.PATH_TO_CBIR_MODEL, 'rb'))
if args['keywords']:
    train_model.generate_products_keywords()
    products = pd.read_pickle(config.PATH_TO_PRODUCT_DATASET)
if args['image']:
    print(perform_search(args['image']))
