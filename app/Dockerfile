FROM openvino/workbench as workbench
USER root

RUN python3 -m pip install --upgrade pip
RUN apt-get update

WORKDIR /opt/intel/openvino/deployment_tools/tools/model_downloader
RUN pip install -r requirements.in
RUN ./downloader.py --name text-spotting-0002-detector --output_dir /app/models
RUN ./downloader.py --name text-spotting-0002-recognizer-decoder --output_dir /app/models
RUN ./downloader.py --name text-spotting-0002-recognizer-encoder --output_dir /app/models

FROM openvino/ubuntu20_runtime:latest as goldapple
USER root

RUN python3 -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  libgtk-3.0 \
  curl \
  unzip \
  libsm6 \
  libxext6 \
  libgl1-mesa-glx \
  build-essential \
  gcc && \
  apt autoremove -y && \
  rm -rf /var/lib/apt/lists/*

COPY . /app/
COPY --from=workbench  /app/models /app/models

WORKDIR /root/.cache/torch/hub/checkpoints/
RUN curl -O 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

WORKDIR /app
RUN pip install -r requirements.txt
RUN chmod +x  search.py

RUN curl -OL 'https://github.com/karina-rev/cosmetic_recognition/releases/download/1.0/output.zip' \
&& unzip output.zip \
&& rm output.zip


VOLUME ["/app/products", "/app/logs", "/app/output"]
EXPOSE 8080
RUN /bin/bash -c "source /opt/intel/openvino/bin/setupvars.sh"
CMD ["/bin/bash"]


