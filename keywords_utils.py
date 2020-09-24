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


def get_keywords(image):
    """
    Обработка изображения и поиск слов с помощью pytesseract
    :param image: изображение для поиска слов
    :return: множество set найденных слов
    """
    custom_config = r'--oem 3 --psm 3 -l fra+eng+rus+ita'
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    keywords = pytesseract.image_to_string(image_bgr, config=custom_config).split()

    clear_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clear_img = cv2.medianBlur(clear_img, 3)
    clear_img = cv2.Canny(clear_img, 100, 200)

    keywords = keywords + pytesseract.image_to_string(clear_img, config=custom_config).split()

    clear_img = deskew(clear_img)

    keywords = keywords + pytesseract.image_to_string(clear_img, config=custom_config).split()

    clear_img = Image.open(set_image_dpi(image))
    clear_img = cv2.cvtColor(np.array(clear_img), cv2.COLOR_RGB2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    clear_img = cv2.dilate(clear_img, kernel, iterations=1)
    clear_img = cv2.erode(clear_img, kernel, iterations=1)
    cv2.adaptiveThreshold(cv2.bilateralFilter(clear_img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv2.THRESH_BINARY, 31, 2)

    keywords = keywords + pytesseract.image_to_string(clear_img, config=custom_config).split()
    keywords = [k.translate(str.maketrans('', '', punctuation)) for k in keywords]
    keywords = set([k.lower() for k in keywords if len(k) > 2])
    return keywords
