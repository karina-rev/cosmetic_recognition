FROM openvino-dev:2020.4.287 as goldapple

RUN apt-get update && apt-get install -y --no-install-recommends \
  libgtk-3.0 \
  cmake \
  python3 \
  python3-pip \
  python3-setuptools \
  curl \
  unzip \
  ffmpeg \
  libsm6 \
  libpq-dev \
  libxext6 \
  libpython3.6 \
  build-essential \
  python3-dev \
  gcc \
  sudo && \
  apt autoremove -y && \
  rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

COPY . /app/

WORKDIR /root/.cache/torch/hub/checkpoints/
RUN curl -O 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

WORKDIR /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/
RUN pip install -r requirements.in
RUN ./downloader.py --name text-spotting-0002-detector --output_dir /app/models
RUN ./downloader.py --name text-spotting-0002-recognizer-decoder --output_dir /app/models
RUN ./downloader.py --name text-spotting-0002-recognizer-encoder --output_dir /app/models

WORKDIR /app
RUN pip install -r requirements.txt
RUN chmod +x  search.py

RUN curl -OL 'https://github.com/karina-rev/cosmetic_recognition/releases/download/1.0/output.zip' \
&& unzip output.zip \
&& rm output.zip

CMD ["bash"]

