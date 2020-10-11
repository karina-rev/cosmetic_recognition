# cosmetic_recognition

Распознование наименования продукта с помощью:
1) Cистемы Content-Based Image Retrieval с использованием предобученной модели ResNet и Faiss индексации
2) Optical character recognition c использованием библиотеки OpenVINO (предобученная модель text-spotting)

<h5>Структура:</h5>  
<b>app/config.py</b> - конфигурационные переменные  <br>
<b>app/search.py</b> - главный скрипт; поиск названия продукта по изображению  <br>
<b>app/text_spotting.py</b> - использование предобученных моделей text-spotting для извлечения слов из изображения <br>
<b>app/train_model.py</b> - функции для обучения модели  <br>
<b>app/utils.py</b> - вспомогательные функции <br>

<h5>Директории: </h5> 
(находятся в zip-архивах в <a href='https://github.com/karina-rev/cosmetic_recognition/releases/tag/1.0'>релизе</a>) <br>
<b>output</b> - обученные модели. <br>
 -- <b>faiss.index</b> - индекс faiss  <br>
 -- <b>products.pkl</b> - датафрейм, распарсенный из данных из xml со словами, которые удалось найти на каждом изображении  <br>

<br>
<h5>Докер</h5>
<code>docker-compose build</code> 
Создается 3 образа: <br><br>
1) ubuntu:18.04 <br>
2) openvino-dev:2020.4.287 <br>
3) goldapple <br><br>
Запуск скрипта с помощью докера: <br>
<code>IMAGE=IMAGE_URL docker-compose run goldapple</code>
<br><br>
<a href='https://github.com/karina-rev/cosmetic_recognition/releases/download/1.0/test.txt'>Несколько примеров ссылок с изображениями, которые правильно классифицируются</a>
