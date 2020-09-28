# cosmetic_recognition

Распознование наименования продукта с помощью:
1) Cистемы Content-Based Image Retrieval с использованием предобученной модели ResNet и Faiss индексации
2) Optical character recognition c использованием библиотеки pytesseract

<h5>Структура:</h5>  
<b>cbir_model.py</b> - класс для работы с моделью CBIR  <br>
<b>config.py</b> - конфигурационные переменные  <br>
<b>keywords_utils.py</b> - функции для извлечения слов из изображения  <br>
<b>search.py</b> - главный скрипт; поиск названия продукта по изображению  <br>
<b>train_model.py</b> - функции для обучения модели  <br>
<b>utils.py</b> - вспомогательные функции для чтения изображений  <br>

<h5>Директории: </h5> 
(находятся в zip-архивах в релизе https://github.com/karina-rev/cosmetic_recognition/releases/tag/1.0) <br>
<b>output</b> - здесь хранятся обученные модели. <br>
 -- <b>cbir.pkl</b> - модель CBIR из файла cbir_model.py  <br>
 -- <b>faiss.index</b> - индекс faiss  <br>
 -- <b>product_keywords.pkl</b> - датафрейм с путями до каждого изображения, его наименованием и словами, которые удалось найти на каждом изображении  <br>
<b>products</b> - база с продуктами  <br>
<b>test</b> - несколько изображений из интернета для тестирования  <br>
<br><br>
Запускается скрипт командой: <code>python3 search.py --image 'test/1.jpg'</code> <br>
Если при этом в папке output не лежат обученные модели, то сначала они создаются и потом производится поиск. <br>
Можно отдельно запустить переобучение CBIR (создаются/перезаписываются файлы cbir.pkl и faiss.index) или OCR 
(создается/перезаписывается файл product_keywords.pkl) модели: <br>
<code>python3 search.py --cbir</code> <br>
<code>python3 search.py --keywords</code> 

<h5>Докер</h5>
Запуск скрипта с помощью докера: <br>
<code>docker container run -it [docker_image] --image 'test/[test_cosmetic_image_number].jpg'</code>
<br><br>
<a href='https://github.com/karina-rev/cosmetic_recognition/releases/download/1.0/test_real_names.txt'>Правильные названия продуктов для тестирования</a>
