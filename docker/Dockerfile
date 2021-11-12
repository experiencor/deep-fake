# This is a sample Dockerfile
# You may adapt this to meet your environment needs
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libmagic-dev git -y

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps pytorch_lightning==1.4.2
RUN git clone https://github.com/open-mmlab/mmaction2.git && cd mmaction2 && pip install --no-cache-dir -r requirements/build.txt && pip install --no-cache-dir -v -e .
RUN pip install --no-cache-dir mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install --no-cache-dir --pre --force-reinstall torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
RUN pip install --no-cache-dir numpy==1.20 torchmetrics==0.6.0 fsspec==2021.11.0 \
    tensorboard==2.7.0 pyDeprecate==0.3.1 void==0.1.3

COPY retinaface retinaface
RUN cd retinaface && python setup.py install

COPY SyncNetModel.py SyncNetModel.py
COPY model.ckpt model.ckpt
COPY crop.py crop.py
COPY model.py model.py
COPY dataset.py dataset.py
COPY data_loader.py data_loader.py
COPY utils.py utils.py
COPY eval.py eval.py
COPY config.json config.json
COPY syncnet_v2.model syncnet_v2.model

ENTRYPOINT ["python","/app/eval.py"]