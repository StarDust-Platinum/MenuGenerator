FROM python:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8888

ENTRYPOINT ["streamlit", "run", "src/main.py", "--server.port=8888", "--server.address=0.0.0.0"]