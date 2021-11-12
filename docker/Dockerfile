# This is a sample Dockerfile
# You may adapt this to meet your environment needs
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY model.ckpt model.ckpt
COPY crop.py crop.py
COPY model.py model.py
COPY dataset.py dataset.py
COPY data_loader.py data_loader.py
COPY model.py model.py
COPY eval.py eval.py

ENTRYPOINT ["python","/app/eval.py"]