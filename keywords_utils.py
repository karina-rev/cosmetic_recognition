import string
import cv2
import numpy as np
import tempfile
import pytesseract
from PIL import Image

punctuation = string.punctuation + '“—'


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def set_image_dpi(im):
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def image_to_text(img):
    custom_config = r'--oem 3 --psm 3 -l fra+eng+rus+ita'
    return pytesseract.image_to_string(img, config=custom_config).split()


def get_keywords(img):
    """
    Обработка изображения и поиск слов с помощью pytesseract
    :param image: изображение для поиска слов
    :return: множество set найденных слов
    """
    image_0 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    image_1 = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
    image_1 = cv2.medianBlur(image_1, 3)
    image_1 = cv2.Canny(image_1, 100, 200)

    image_2 = deskew(image_1)

    image_3 = Image.open(set_image_dpi(img))
    image_3 = cv2.cvtColor(np.array(image_3), cv2.COLOR_RGB2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    image_3 = cv2.dilate(image_3, kernel, iterations=1)
    image_3 = cv2.erode(image_3, kernel, iterations=1)
    cv2.adaptiveThreshold(cv2.bilateralFilter(image_3, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv2.THRESH_BINARY, 31, 2)

    images = [image_0, image_1, image_2, image_3]

    keywords = []
    for image in images:
        keywords += image_to_text(image)

    keywords = [k.translate(str.maketrans('', '', punctuation)) for k in keywords]
    keywords = set([k.lower() for k in keywords if len(k) > 2])
    return keywords
