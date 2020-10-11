import pandas as pd
import numpy as np
import utils
import config
import faiss
import requests
import os
import torch
import xml.etree.ElementTree as ET
import pickle5 as pickle
from PIL import Image
from torchvision import models
from torch.utils.data import Dataset
from io import BytesIO
import text_spotting

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProductsDataset(Dataset):
    """Products dataset."""

    def __init__(self, products_path, transform=None):
        with open(products_path, 'rb') as f:
            self.products = pickle.load(f)
        self.transform = transform
        self.http = requests.Session()
        self.http.mount("https://", utils.adapter)

    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        picture_url = self.products.iloc[idx]['picture']
        response = self.http.get(picture_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def make_products_dataframe():
    """
    Чтение, создание датасета с продуктами. Сохранение в config.PATH_TO_XML_DATABASE
    """
    print('Making products dataframe...')
    http = requests.Session()
    http.mount("https://", utils.adapter)
    response = http.get(config.URL_TO_XML_FILE)

    root = ET.fromstring(response.content)
    df_cols = ['url', 'price', 'oldprice', 'currencyId', 'categoryId', 'picture', 'typePrefix', 'model',
               'vendor', 'barcode', 'description']
    rows = []
    for node in root.findall('shop/offers/offer'):
        res = []
        for col in df_cols:
            if node is not None and node.find(col) is not None:
                res.append(node.find(col).text)
            else:
                res.append(None)
        rows.append({df_cols[i]: res[i] for i, _ in enumerate(df_cols)})

    print('Writing products dataframe')
    products = pd.DataFrame(rows, columns=df_cols)
    products.to_pickle(config.PATH_TO_PRODUCT_DATASET)


def train_faiss_index():
    print('Training faiss index...')
    if not os.path.exists(config.PATH_TO_PRODUCT_DATASET):
        make_products_dataframe()

    dataset = ProductsDataset(config.PATH_TO_PRODUCT_DATASET, utils.transform_torch())
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
            if i % 5000 == 0:
                print(f'Rows processed: {i}')

    # создаем Flat индекс и добавляем векторы дескрипторов
    print('Writing faiss index')
    index = faiss.IndexFlatL2(2048)
    descriptors = np.vstack(descriptors)
    index.add(descriptors)
    faiss.write_index(index, config.PATH_TO_FAISS_INDEX)


def train_products_keywords():
    """
    Поиск с помощью OCR слов на каждом изобрадении в базе
    Добавление найденных слов в столбец keywords в датасет с продуктами
    """
    print('Training products keywords...')
    if not os.path.exists(config.PATH_TO_PRODUCT_DATASET):
        make_products_dataframe()

    with open(config.PATH_TO_PRODUCT_DATASET, 'rb') as f:
        products = pickle.load(f)
    products['keywords'] = products['vendor'] + ' ' + products['model']
    products['keywords'] = products['keywords'].str.replace('мл', '').str.lower().str.split()

    http = requests.Session()
    http.mount("https://", utils.adapter)

    i = 0
    for index, row in products.iterrows():
        response = http.get(row['picture'])
        image = Image.open(BytesIO(response.content)).convert('RGB')
        row['keywords'] = row['keywords'] + text_spotting.search_text(image)

        i += 1
        if i % 5000 == 0:
            print(f'Rows processed: {i}')

    products.to_pickle(config.PATH_TO_PRODUCT_DATASET)

