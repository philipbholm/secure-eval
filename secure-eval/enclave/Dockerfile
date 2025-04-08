FROM python:3.11.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY certs/ ./certs/

COPY server.py nsm.py ./

CMD ["/usr/local/bin/python3", "/app/server.py"]