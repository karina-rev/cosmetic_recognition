import pickle
import glob
import pandas as pd
import keywords_utils
import utils
import config
from PIL import Image
from cbir_model import CBIR


def train_cbir_model():
    """
    Создание модели CBIR, файл сохраняется в output/cbir.pkl
    """
    print('Training and saving CBIR model...')
    with open(config.PATH_TO_CBIR_MODEL, 'wb') as output:
        cbir = CBIR().train()
        pickle.dump(cbir, output, pickle.HIGHEST_PROTOCOL)


def generate_products_keywords():
    """
    Поиск с помощью OCR слов на каждом изобрадении в базе
    Создание датасета products, файл output/product_keywords.pkl
    Products содержит в себе path - путь к изображению продукта, name - название продукта, keywords - найденные слова
    """
    print('Generating products keywords...')
    product_files = glob.glob(config.PATH_TO_FILES + '/*/*')
    product_names = []
    product_paths = []
    product_keywords = []

    for product_file in product_files:
        name = product_file.split('/')[1]
        path = product_file.split('/', 1)[1]

        if path.endswith('.heic'):
            image = utils.read_heic_img(config.PATH_TO_FILES + '/' + path)
        else:
            image = Image.open(config.PATH_TO_FILES + '/' + path)

        keywords = keywords_utils.get_keywords(image)
        name_keywords = set(name.lower().split())

        product_names.append(name)
        product_paths.append(path)
        product_keywords.append(name_keywords | keywords)

    products = pd.DataFrame(data={'path': product_paths,
                                  'name': product_names,
                                  'keywords': product_keywords})

    products.to_pickle(config.PATH_TO_PRODUCT_DATASET)


def add_products_keywords(path_to_new_files):
    """
    Добавление в датасет products новых продуктов, сохранение файла в output/product_keywords.pkl
    :param path_to_new_files: список путей новых продуктов
    """
    try:
        products = pd.read_pickle(config.PATH_TO_PRODUCT_DATASET)
    except FileNotFoundError:
        generate_products_keywords()
        return

    for path_to_file in path_to_new_files:
        name = path_to_file.split('/')[0]
        path = path_to_file

        if path.endswith('.heic'):
            image = utils.read_heic_img(config.PATH_TO_FILES + '/' + path)
        else:
            image = Image.open(config.PATH_TO_FILES + '/' + path)

        keywords = keywords_utils.get_keywords(image)
        name_keywords = set(name.lower().split())

        products = products.append({'path': path,
                                    'name': name,
                                    'keywords': name_keywords | keywords}, ignore_index=True)

    products.to_pickle(config.PATH_TO_PRODUCT_DATASET)
