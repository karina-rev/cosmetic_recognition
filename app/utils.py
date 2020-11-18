import pyheif
import config
import time
from PIL import Image
from io import BytesIO
from torchvision import models, transforms
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

retry_strategy = Retry(
    total=5,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"])
adapter = HTTPAdapter(max_retries=retry_strategy)
MODEL = models.resnet101(pretrained=True)


def pooling_output(x):
    for layer_name, layer in MODEL._modules.items():
        x = layer(x)
        if layer_name == 'avgpool':
            break
    return x


def transform_torch():
    return transforms.Compose([
        transforms.Resize(size=[224, 224], interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def read_heic_img(path):
    heif_file = pyheif.read(path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    return image


def get_image(path_to_image, http_session):
    if path_to_image.startswith('http'):
        response = http_session.get(path_to_image)
        image = Image.open(BytesIO(response.content))
    else:
        image = open_local_image(path_to_image)
    return image


def open_local_image(path_to_image):
    path_to_image = config.PATH_TO_PRODUCT_FOLDER + path_to_image
    if path_to_image.endswith('.heic') or path_to_image.endswith('.HEIC'):
        image = read_heic_img(path_to_image)
    else:
        image = Image.open(path_to_image)
    return image
