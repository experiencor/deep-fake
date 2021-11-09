# This is a sample Dockerfile
# You may adapt this to meet your environment needs
FROM python:3
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ruby.ckpt model.ckpt

COPY custom_labeled_dataset.py custom_labeled_dataset.py
COPY crop.py crop.py
COPY eval.py eval.py

ENTRYPOINT ["python","/app/eval.py"]