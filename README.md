# cosmetic_recognition

Текущие модели: https://disk.yandex.ru/d/4KFXLA1Eut2_-w <br>
1. Собрать докер: 
```bash
docker-compose build
``` 
2. Разархивировать в папку app/output модельки
3. Запустить докер:
```bash
docker-compose up -d
``` 
4. Немного подождать, пока загрузится сервер
5. Загрузить клиента:
```bash
docker run -it -v путь_до_папки_проекта_cosmetic_recognition/client/data:/data --network=cosmetic_recognition_default cosmetic_client
```
6. В клиентском контейнере:
```bash
python3 client.py -i 'ссылка_на_изображение'
```
```bash
python3 client.py -i 'data/название_изображения'
```
7. Первые изображения распознаются долго, дальше в среднем 3-4 секунды.
8. Время распознавания можно посмотреть в логе на сервере: logs/cosmetic_recognition.log

Распознование наименования продукта по изображению с помощью:
1) Cистемы Content-Based Image Retrieval с использованием предобученной модели ResNet и Faiss индексации
2) Optical character recognition c использованием библиотеки OpenVINO (предобученная модель text-spotting)

## Структура:
<b>app/config.py</b> - конфигурационные переменные  <br>
<b>app/main.py</b> - главный скрипт, запуск приложения <br>
<b>app/search.py</b> - поиск названия продукта по изображению  <br>
<b>app/text_spotting.py</b> - использование предобученных моделей text-spotting для извлечения слов из изображения <br>
<b>app/train_model.py</b> - функции для обучения и обновления модели  <br>
<b>app/utils.py</b> - вспомогательные функции <br>
<br>
<b>client/client.py</b> - скрипт для тестирования приложения<br>


## Директории:
(находятся в zip-архивах в <a href='https://github.com/karina-rev/cosmetic_recognition/releases/tag/1.0'>релизе</a>) <br>
<b>output</b> - обученные модели. <br>
 -- <b>faiss.index</b> - индекс faiss  <br>
 -- <b>products.pkl</b> - датафрейм, распарсенный из данных из xml со словами, которые удалось найти на каждом изображении  <br>

## Докер
<h4>В директории app:</h4> 

```bash
docker-compose up -d 
```
Создается образ cosmetic_recognition. Запускается сервер в контейнере cosmetic_recognition, сеть cosmetic_recognition_default, порт 8080. Для получения результата необходимо отправить POST-запрос с байтами изображения на cosmetic_recognition:8080/image. <br>
Также, на сервере хранится лог /logs/cosmetic_recognition.log

<h6>Обучение/обновление модели. </h6>
Для создания новой модели (faiss.index + products.pkl) или обновления текущей на сервере cosmetic_recognition необходимо запустить скрипт train_model.py <br><br>
<b>Cоздание новой модели: </b>

```bash 
python3 train_model -c 
``` 
<b>Обновление текущей модели:</b> удаление продуктов, которых нет в базе на данный момент; обновление значений/изображений продуктов; добавление новых, в том числе добавление дополнительных изображений продуктов из директории. Все изображения, которые необходимо поместить в модель, должны находиться по пути: cosmetic_recognition/products.

```bash
python3 train_model -u
```

<h4>В директории client:</h4> 

```bash
docker build -t cosmetic_client .
``` 
Создается образ cosmetic_client. С помощью данного образа можно отправлять изображения на cosmetic_recognition. Изображение можно загружать с помощью внешней ссылки или загрузить в папку client/data. Запуск контейнера командой:

```bash
docker run -it -v путь_до_папки_проекта_cosmetic_recognition/client/data:/data --network=cosmetic_recognition_default cosmetic_client
```

Далее в запустившемся контейнере: <br>
```bash
python3 client.py -i 'ссылка_на_изображение'
```
```bash
python3 client.py -i 'data/название_изображения'
```
<a href='https://github.com/karina-rev/cosmetic_recognition/releases/download/1.0/test.txt'>Несколько примеров ссылок с изображениями, которые правильно классифицируются</a>
