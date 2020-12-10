FROM python:3.7.4

RUN apt-get update
RUN apt-get -y  upgrade
#RUN apt-get -y install python3-pip
WORKDIR /opt
COPY . /opt
COPY opt/ /opt/
COPY opt/data/ /opt/
COPY opt/models/ /opt/
COPY opt/training_data/ /opt/
RUN pip3 --no-cache-dir install tensorflow==1.14.0
RUN pip3 --no-cache-dir install fastapi
RUN pip3 --no-cache-dir install sklearn
RUN pip3 --no-cache-dir install scikit-learn
RUN pip3 --no-cache-dir install sklearn_crfsuite
RUN pip3 --no-cache-dir install nltk
RUN pip3 --no-cache-dir install spacy
RUN pip3 --no-cache-dir install tornado
RUN pip3 --no-cache-dir install h5py==2.10.0
RUN pip3 --no-cache-dir install keras==2.3.1
RUN pip3 --no-cache-dir install sklearn_crfsuite
RUN pip3 --no-cache-dir install sklearn
RUN pip3 --no-cache-dir install elasticsearch
RUN pip3 --no-cache-dir install uvicorn
RUN pip3 --no-cache-dir install aiohttp
RUN python3 -m spacy download en_core_web_md
RUN python3 -m nltk.downloader punkt
EXPOSE 8000
ENTRYPOINT ["python3","fastapi_server.py"]
CMD ["--port 8000", "--host 0.0.0.0"]

 


