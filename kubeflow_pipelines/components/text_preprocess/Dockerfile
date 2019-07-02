FROM pytorch/pytorch:latest
RUN pip install tensorflow-gpu>=1.10.0,<=1.14.0 umap-learn visdom \
webrtcvad librosa>=0.5.1 matplotlib>=2.0.2 numpy>=1.14.0 \
scipy>=1.0.0 tqdm sounddevice Unidecode inflect PyQt5 \
multiprocess numba
ARG url
RUN git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning
WORKDIR Real_Time_Voice_Cloning
RUN wget -O preprocess.py  $url
ENTRYPOINT python preprocess.py 