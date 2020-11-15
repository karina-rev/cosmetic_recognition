# cosmetic_recognition

Распознование наименования продукта с помощью:
1) Cистемы Content-Based Image Retrieval с использованием предобученной модели ResNet и Faiss индексации
2) Optical character recognition c использованием библиотеки OpenVINO (предобученная модель text-spotting)

<h5>Структура:</h5>  
<b>app/config.py</b> - конфигурационные переменные  <br>
<b>app/main.py</b> - главный скрипт, запуск приложения <br>
<b>app/search.py</b> - поиск названия продукта по изображению  <br>
<b>app/text_spotting.py</b> - использование предобученных моделей text-spotting для извлечения слов из изображения <br>
<b>app/train_model.py</b> - функции для обучения и обновления модели  <br>
<b>app/utils.py</b> - вспомогательные функции <br>
<br>
<b>client/client.py</b> - скрипт для тестирования приложения<br>


<h5>Директории: </h5> 
(находятся в zip-архивах в <a href='https://github.com/karina-rev/cosmetic_recognition/releases/tag/1.0'>релизе</a>) <br>
<b>output</b> - обученные модели. <br>
 -- <b>faiss.index</b> - индекс faiss  <br>
 -- <b>products.pkl</b> - датафрейм, распарсенный из данных из xml со словами, которые удалось найти на каждом изображении  <br>

<br>
<h5>Докер</h5>
В директории app: <br>
<code>docker-compose up -d</code> <br>
Создается образ cosmetic_recognition. Запускается сервер в контейнере cosmetic_recognition, сеть cosmetic_recognition_default, порт 8080. Для получения результата необходимо отправить POST-запрос с байтами изображения на cosmetic_recognition:8080/image. <br>
Также, на сервере хранится лог /logs/cosmetic_recognition.log
<br><br>
<b>Обучение/обновление модели. </b> <br>
Для создания новой модели (faiss.index + products.pkl) или обновления текущей на сервере cosmetic_recognition необходимо запустить скрипт train_model.py <br>
<code>python3 train_model -c</code>  -  создание новой модели
<code>python3 train_model -u</code>  -  обновление текущей модели: удаление продуктов, которых нет в базе на данный момент; обновление значений/изображений продуктов; добавление новых, в том числе добавление дополнительных изображений продуктов из директории. Все изображения, которые необходимо поместить в модель, должны находиться по пути: cosmetic_recognition/products. 
  <br><br>
В директории client: <br>
<code>docker build -t cosmetic_client .<code> <br>
Создается образ cosmetic_client. С помощью данного образа можно отправлять изображения на cosmetic_recognition. Изображение можно загружать с помощью внешней ссылки или загрузить в папку client/data. Запуск контейнера командой: <br>
<code>docker run -it -v путь_до_папки_проекта_cosmetic_recognition/client/data:/data --network=cosmetic_recognition_default cosmetic_client</code> <br>
Далее в запустившемся контейнере: <br>
 <code>python3 client.py -i 'ссылка_на_изображение'</code> или <code>python3 client.py -i 'data/название_изображения'</code> 
<br>
<a href='https://github.com/karina-rev/cosmetic_recognition/releases/download/1.0/test.txt'>Несколько примеров ссылок с изображениями, которые правильно классифицируются</a>
