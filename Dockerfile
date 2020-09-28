FROM python:3

# set a directory for the app

WORKDIR /

# copy all the files to the container
COPY . .

# install dependencies
RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y 
RUN pip install -r requirements.txt
RUN chmod +x  search.py

RUN curl -OL 'https://github.com/karina-rev/cosmetic_recognition/releases/download/1.0/output.zip' \
&& unzip output.zip \
&& rm output.zip

RUN curl -OL 'https://github.com/karina-rev/cosmetic_recognition/releases/download/1.0/products.zip' \
&& unzip products.zip \
&& rm products.zip

RUN curl -OL 'https://github.com/karina-rev/cosmetic_recognition/releases/download/1.0/test.zip' \
&& unzip test.zip \
&& rm test.zip

# run the command
ENTRYPOINT ["python3", "search.py"]