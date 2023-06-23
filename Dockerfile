FROM python:3.9
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
RUN pip3 install protobuf==3.20
CMD ["streamlit", "run", "main.py"]

