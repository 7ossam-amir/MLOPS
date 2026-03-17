FROM python:3.10-slim

WORKDIR /APP

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

//add timeout

COPY . .

CMD ["python", "main.py"]