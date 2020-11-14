import pandas as pd
import numpy as np
import utils
import glob
import config
import faiss
import requests
import torch
import logging
import argparse
import xml.etree.ElementTree as ET
from torchvision import models
from torch.utils.data import Dataset
from text_spotting import TextSpotting


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DF_COLS = ['url', 'price', 'oldprice', 'currencyId', 'categoryId', 'picture', 'typePrefix', 'model',
           'vendor', 'barcode', 'description']
logging.basicConfig(filename=config.PATH_TO_LOG_FILE,
                    format='%(asctime)s [%(levelname)s] - %(message)s',
                    level=logging.INFO)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--create", action='store_true', help="create new models")
ap.add_argument("-u", "--update", action='store_true', help="update models")
args = vars(ap.parse_args())


class ProductsDataset(Dataset):
    """Products dataset."""

    def __init__(self, products, transform=None):
        self.products = products
        self.transform = transform
        self.http = requests.Session()
        self.http.mount("https://", utils.adapter)

    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        picture_url = self.products.iloc[idx]['picture']
        image = utils.get_image(picture_url, self.http)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def create_models():
    """
    Создание и обучение моделей. Сохранение модели с обученными индексами в config.PATH_TO_FAISS_INDEX,
    сохранение датафрейма с найденными словами на изображении в config.PATH_TO_PRODUCT_DATASET
    """
    logging.info('Создание моделей')
    # создание и сохранение датафрейма с продуктами
    make_products_dataframe()
    products = pd.read_pickle(config.PATH_TO_PRODUCT_DATASET)

    # создание faiss индекса
    descriptors = train_faiss_descriptors(products)
    logging.info('Запись faiss индекса')
    index = faiss.IndexFlatL2(2048)
    index = faiss.IndexIDMap2(index)
    index.add_with_ids(np.vstack(descriptors), np.hstack([i for i in range(products.shape[0])]))
    faiss.write_index(index, config.PATH_TO_FAISS_INDEX)

    # добавление в датафрейм с продуктами найденных слов по каждому изображению
    products = train_products_keywords(products)
    logging.info('Запись датафрейма в файл')
    products.to_pickle(config.PATH_TO_PRODUCT_DATASET)

    assert index.ntotal == products.shape[0]
    logging.info(f'Модель создана и обучена. '
                 f'Всего {products["vendor_code"].nunique()} продуктов и {products.shape[0]} изображений в общем')


def make_products_dataframe():
    """
    Создание датафрейма с продуктами по xml-файлу и добавление дополнительынх изображений из директории.
    Сохранение в config.PATH_TO_PRODUCT_DATASET
    """
    # создание датафрейма по xml-файлу
    rows = get_data_from_xml()
    products = pd.DataFrame(rows, columns=DF_COLS)
    # артикул продукта
    products['vendor_code'] = products['url'].str.split('utm_term=').str[-1]
    # добавление в датафрейм дополнительных изображений продуктов из директории
    products, _ = add_additional_images(products)

    logging.info('Запись датафрейма в файл')
    products.to_pickle(config.PATH_TO_PRODUCT_DATASET)


def get_data_from_xml():
    """
    Чтение xml-файла по пути config.URL_TO_XML_FILE
    :return: строки с данными
    """
    logging.info(f'Создание датафрейма по xml-файлу по пути {config.URL_TO_XML_FILE}')
    http = requests.Session()
    http.mount("https://", utils.adapter)
    response = http.get(config.URL_TO_XML_FILE)
    root = ET.fromstring(response.content)

    rows = []
    for node in root.findall('shop/offers/offer'):
        res = []
        for col in DF_COLS:
            if node is not None and node.find(col) is not None:
                res.append(node.find(col).text)
            else:
                res.append(None)
        if res[0]:
            rows.append({DF_COLS[i]: res[i] for i, _ in enumerate(DF_COLS)})

    return rows


def add_additional_images(products):
    """
    Добавление в датафрейм изображений продуктов в директории config.PATH_TO_PRODUCT_FOLDER
    :param products: датафрейм с текущими продуктами
    :return: датафрейм products с добавленными строками с дополнительными изображениями продуктов,
    список idx_to_add с индексами добавленных строк для их дальнейшей обработки
    """
    logging.info(f'Добавление в датафрейм изображений из директории {config.PATH_TO_PRODUCT_FOLDER}')
    product_files = glob.glob(config.PATH_TO_PRODUCT_FOLDER + '/*/*')
    # счетчики для отслеживания количества обработанных продуктов
    count_all_products = 0
    count_absent_products = 0
    count_all_images = 0
    count_error_images = 0
    # последний индекс в датафрейме, после которого можно добавлять новые строки
    curr_index = products.index.max()
    # индексы добавленных строк
    idx_to_add = []

    # читаем отдельно каждую папку с отдельным продуктом
    for product_path in product_files:
        count_all_products += 1
        vendor_code = product_path.split('/')[3]
        # проверяем, если в датасете продукт с таким артикулом
        product_to_duplicate = products.loc[products['vendor_code'] == str(vendor_code)]
        if product_to_duplicate.empty:
            count_absent_products += 1
            logging.warning(f'В базе нет продукта с артикулом {vendor_code}')
            continue

        # читаем все изображения текущего продукта
        pictures = glob.glob(product_path + '/*')
        for picture_full_path in pictures:
            count_all_images += 1
            picture_name = picture_full_path.replace(config.PATH_TO_PRODUCT_FOLDER, '')
            # если такое изображение уже есть в датафрейме, читаем следующее
            if picture_name in products['picture'].values:
                continue
            # открываем изображение в зависимости от его расширения
            try:
                utils.open_local_image(picture_full_path)
                # дублируем строку с найденным продуктом в датасете по текущему артикулу
                # меняем только путь к изображению
                curr_index += 1
                products.loc[curr_index] = product_to_duplicate.values[0]
                products.loc[curr_index]['picture'] = picture_name
                idx_to_add.append(curr_index)
            # если изображение не открывается, пишем в лог
            except Exception as ex:
                count_error_images += 1
                logging.warning(f'Не получилось открыть изображение, ошибка: {ex}')

        if count_all_products % 500 == 0:
            logging.debug(f'Обработано строк: {count_all_products}')
    logging.info(f'Из {count_all_products} продуктов в базе не нашлось {count_absent_products}')
    logging.info(
        f'Из {count_all_images} изображений продуктов, которые нашлись в базе, не обработано {count_error_images}')
    return products, idx_to_add


def train_faiss_descriptors(products):
    """
    Создание дескрипторов для faiss индекса
    :param products: датарфейм или часть датафрейма с продуктами, для которых необходимо обучить дескрипторы
    :return: список дескрипторов descriptors
    """
    logging.info('Создание дескрипторов для faiss индекса')
    dataset = ProductsDataset(products, utils.transform_torch())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = models.resnet101(pretrained=True)
    descriptors = []

    model.to(DEVICE)
    i = 0
    with torch.no_grad():
        model.eval()
        for inputs in dataloader:
            result = utils.pooling_output(inputs.to(DEVICE))
            descriptors.append(result.cpu().view(1, -1).numpy())
            torch.cuda.empty_cache()

            i += 1
            if i % 500 == 0:
                logging.info(f'Обработано строк: {i}')
    return descriptors


def train_products_keywords(products):
    """
    Поиск с помощью OpenVINO слов на каждом изображении в базе
    Добавление найденных слов в столбец keywords в датафрейм с продуктами
    :param products: датарфейм или часть датафрейма с продуктами, для которых необходимо получить слова
    :return: датафрейм с найденными словами для каждого изображения в столбце keywords
    """
    logging.info('Поиск и добавление в датафрейм слов по каждому изображению')
    products['keywords'] = products['vendor'] + ' ' + products['model']
    products['keywords'] = products['keywords'].str.replace('мл','').str.lower().str.split()
    http = requests.Session()
    http.mount("https://", utils.adapter)

    # загрузка модели OpenVINO
    text_spotting_net = TextSpotting().train_network()
    i = 0
    for index, row in products.iterrows():
        image = utils.get_image(row['picture'], http)
        image = image.convert('RGB')
        row['keywords'] = row['keywords'] + text_spotting_net.search_text(image)

        i += 1
        if i % 500 == 0:
            logging.info(f'Обработано строк: {i}')

    return products


def update_models():
    """
    Обновление моделей. Сохранение обновленной модели с обученными индексами в config.PATH_TO_FAISS_INDEX,
    сохранение обновленного датафрейма с найденными словами на изображении в config.PATH_TO_PRODUCT_DATASET
    """
    logging.info('Обновление моделей')
    # читаем новую версию базы в current_products
    rows = get_data_from_xml()
    current_products = pd.DataFrame(rows, columns=DF_COLS)
    current_products['vendor_code'] = current_products['url'].str.split('utm_term=').str[-1]
    # читаем текущую обученную версию в products_to_update, её и будем обновлять
    products_to_update = pd.read_pickle(config.PATH_TO_PRODUCT_DATASET)
    index = faiss.read_index(config.PATH_TO_FAISS_INDEX)
    assert index.ntotal == products_to_update.shape[0]

    # удаление из моделей тех продуктов, которых уже нет в xml или уже нет в директории
    products_to_update = delete_products(current_products, products_to_update)
    # обновление значений согласно новому xml
    # если обновились ссылки на изображения, дообучение моделей на новых ссылках
    products_to_update = update_products(current_products, products_to_update)
    # добавление новых продуктов, добавление новых изображений из директории
    # дообучение моделей на новых продуктах/изображениях
    products_to_update = add_products(current_products, products_to_update)

    logging.info(f'Модель обновлена и переобучена. '
                 f'Всего {products_to_update["vendor_code"].nunique()} продуктов и {products_to_update.shape[0]} '
                 f'изображений')


def delete_products(current_products, products_to_update):
    """
    Удаление продуктов, которых нет в новом xml или которые удалили из директории с дополнительными изображениями.
    Удаление индексов продуктов из модели с faiss индексами
    :param current_products: датафрейм из нового xml-файла
    :param products_to_update: текущий датафрейм
    :return: датафрейм products_to_update с удаленными продуктами
    """
    logging.info('Удаление продуктов')
    idx_to_remove = []

    # удаление продуктов, которых нет в новом xml
    # смотрим, какие продукты нужно удалить, исходя из отсутствия артикулов продуктов текущей модели в новом xml
    products_to_update_vendor_code = set(products_to_update['vendor_code'])
    current_products_vendor_code = set(current_products['vendor_code'])
    products_to_update_vendor_code.difference_update(current_products_vendor_code)
    if products_to_update_vendor_code:
        for index, row in products_to_update.iterrows():
            if row['vendor_code'] in products_to_update_vendor_code:
                idx_to_remove.append(index)
                products_to_update.drop(index, inplace=True)

    # удаление строк с изображениями, которых уже нет в директории
    current_product_files = set(glob.glob(config.PATH_TO_PRODUCT_FOLDER + '/*/*/*'))
    product_files_to_update = set(products_to_update[~products_to_update['picture'].str.startswith('http')]['picture'].
                                  values)
    product_files_to_update.difference_update(current_product_files)
    if product_files_to_update:
        for file in product_files_to_update:
            index = products_to_update[products_to_update['picture'] == file].index[0]
            products_to_update.drop(index, inplace=True)
            idx_to_remove.append(index)

    products_to_update.reset_index(inplace=True, drop=True)

    # если ничего не удаляли, возвращаем неизмененный датафрейм
    if not idx_to_remove:
        logging.info('Продуктов для удаления нет')
        return products_to_update

    # удаление из faiss соответствующих векторов
    logging.info(f'Удаление индексов из faiss [{len(idx_to_remove)} строк]')
    index = faiss.read_index(config.PATH_TO_FAISS_INDEX)
    vectors = [index.reconstruct(i) for i in range(index.ntotal)]
    vectors_without_removed = [vectors[i] for i in range(len(vectors)) if i not in idx_to_remove]

    updated_index = faiss.IndexFlatL2(2048)
    updated_index = faiss.IndexIDMap2(updated_index)
    updated_index.add_with_ids(np.vstack(vectors_without_removed),
                               np.hstack([i for i in range(len(vectors_without_removed))]))

    logging.info('Запись')
    faiss.write_index(updated_index, config.PATH_TO_FAISS_INDEX)
    products_to_update.to_pickle(config.PATH_TO_PRODUCT_DATASET)
    assert updated_index.ntotal == products_to_update.shape[0]

    logging.info(f'Удалено {len(idx_to_remove)} строк')
    return products_to_update


def update_products(current_products, products_to_update):
    """
    Обновление значений столбцов текущего датафрейма согласно новому xml-файлу.
    Если обновились ссылки на изображения, обучаем дескрипторы для индексов faiss и находим слова на новых
    изображениях
    :param current_products: датафрейм из нового xml-файла
    :param products_to_update: текущий датафрейм
    :return: датафрейм products_to_update с новыми значениями столбцов
    """
    logging.info('Обновление значений столбцов датафрейма с продуктами')

    count_updated_products = 0
    idx_to_update_picture = []
    # сравниваем построчно значения столбцов текущего датафрейма с новым
    for index_update, row_update in products_to_update.iterrows():
        row_curr = current_products[current_products['vendor_code'] == row_update['vendor_code']]
        for column in DF_COLS:
            if row_update[column] != row_curr[column].values[0]:
                # если изменилось изображение, сохраняем индекс строки для дальнейшей обработки
                if column == 'picture':
                    if row_update[column].startswith('http'):
                        idx_to_update_picture.append(index_update)
                    else:
                        continue
                products_to_update.loc[index_update, column] = row_curr[column].values[0]
                count_updated_products += 1

    # если изображения не обновлялись, возвращаем датафрейм
    if not idx_to_update_picture:
        return products_to_update

    logging.info(f'Добавление и запись новых продуктов по частям '
                 f'[{np.ceil(len(idx_to_update_picture) / 1000).astype("int")}] частей')

    # находим слова на новых изображениях
    logging.info(f'Поиск слов для обновленных изображений [{len(idx_to_update_picture)} строк]')
    products_to_update.loc[idx_to_update_picture] = train_products_keywords(
        products_to_update.loc[idx_to_update_picture])

    # если изображения обновлялись, удаляем векторы старых изображений, обучаем и добавляем новые
    logging.info(f'Обновление дескрипторов faiss [{len(idx_to_update_picture)} строк]')
    index = faiss.read_index(config.PATH_TO_FAISS_INDEX)
    index.remove_ids(np.array(idx_to_update_picture))
    descriptors = train_faiss_descriptors(products_to_update.loc[idx_to_update_picture])
    index.add_with_ids(np.vstack(descriptors), np.hstack(idx_to_update_picture))

    logging.info('Запись')
    faiss.write_index(index, config.PATH_TO_FAISS_INDEX)
    products_to_update.to_pickle(config.PATH_TO_PRODUCT_DATASET)
    assert index.ntotal == products_to_update.shape[0]

    logging.info(f'Изменено {count_updated_products} строк')
    logging.info(f'Переобучена модель для {len(idx_to_update_picture)} изображений')
    return products_to_update


def add_products(current_products, products_to_update):
    """
    Добавление новых продуктов согласно новому xml-файлу, добавление новых изображений из директории с
    дополнительными изображениями. Обучение новых дескрипторов для faiss модели, поиск слов на новых изображениях
    :param current_products: датафрейм из нового xml-файла
    :param products_to_update: текущий датафрейм
    :return: датафрейм products_to_update с новыми продуктами/изображениями
    """
    logging.info('Добавление новых продуктов')

    # добавление новых продуктов из xml-файла
    # смотрим, какие продукты нужно добавить, исходя из отсутствия артикулов продуктов нового xml в текущем датафрейме
    products_to_update_vendor_code = set(products_to_update['vendor_code'])
    current_products_vendor_code = set(current_products['vendor_code'])
    current_products_vendor_code.difference_update(products_to_update_vendor_code)
    idx_to_add_from_xml = []
    if current_products_vendor_code:
        curr_index = products_to_update.index.max()
        for new_product_vendor_code in current_products_vendor_code:
            curr_index += 1
            new_product = current_products.loc[current_products['vendor_code'] ==
                                               new_product_vendor_code].values.tolist()[0]
            # добавим пустую строку для столбца 'keywords', так как в новом датафрейме его еще не существует
            new_product.insert(-1, '')
            products_to_update.loc[curr_index] = new_product
            idx_to_add_from_xml.append(curr_index)

    # добавление новых изображений из директории
    products_to_update, idx_to_add_from_folder = add_additional_images(products_to_update)
    all_idx_to_add = idx_to_add_from_xml + idx_to_add_from_folder

    # если нет новых продуктов и изображений, возвращаем неизмененный датафрейм
    if not all_idx_to_add:
        logging.info('Продуктов для добавления нет')
        return products_to_update

    # поиск слов для новых продуктов/изображений
    logging.info(f'Поиск слов для новых продуктов/изображений [{len(all_idx_to_add)} строк]')
    products_to_update.loc[all_idx_to_add] = train_products_keywords(
        products_to_update.loc[all_idx_to_add])

    # обновляем faiss, обучаем новые дескрипторы
    logging.info(f'Добавление новых дескрипторов в faiss модель [{len(all_idx_to_add)} строк]')
    index = faiss.read_index(config.PATH_TO_FAISS_INDEX)
    descriptors = train_faiss_descriptors(products_to_update.loc[all_idx_to_add])
    index.add_with_ids(np.vstack(descriptors), np.hstack(all_idx_to_add))

    logging.info(f'Запись')
    faiss.write_index(index, config.PATH_TO_FAISS_INDEX)
    products_to_update.to_pickle(config.PATH_TO_PRODUCT_DATASET)
    assert index.ntotal == products_to_update.shape[0]

    if idx_to_add_from_xml:
        logging.info(f'Добавлено {len(idx_to_add_from_xml)} новых продуктов')
    if idx_to_add_from_folder:
        logging.info(f'Добавлено {len(idx_to_add_from_folder)} новых изображений')
    return products_to_update


if __name__ == '__main__':
    if args['create']:
        try:
            create_models()
        except AssertionError:
            logging.error('Assertion Error - Размерности моделей faiss и products не равны')
        except Exception as ex:
            logging.error(str(ex))
    if args['update']:
        try:
            update_models()
        except AssertionError:
            logging.error('Assertion Error - Размерности моделей faiss и products не равны')
        except Exception as ex:
            logging.error(str(ex))