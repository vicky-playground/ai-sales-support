FROM python:3.10

WORKDIR /ai-sales-support
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
