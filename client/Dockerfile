FROM python:3
WORKDIR /
COPY . .

RUN apt-get update
RUN pip install -r requirements.txt

EXPOSE 8070
CMD ["/bin/bash"]

