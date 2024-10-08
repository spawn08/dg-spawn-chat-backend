FROM python:3.6

RUN apt-get update
RUN apt-get -y  upgrade
#RUN apt-get -y install python3-pip
WORKDIR /opt
COPY . /opt
COPY opt/ /opt/
COPY opt/data/ /opt/
COPY opt/models/ /opt/
COPY opt/training_data/ /opt/
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_md
RUN python3 -m nltk.downloader punkt
EXPOSE 8000
ENTRYPOINT ["python3","run.py"]
CMD ["--port 8000", "--host 0.0.0.0"]

 


