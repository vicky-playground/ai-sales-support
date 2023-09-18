FROM python:3.10

WORKDIR /ai-sales-support

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "/ai-sales-support/main.py"]
