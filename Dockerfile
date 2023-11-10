FROM python:3.11

EXPOSE 7860
WORKDIR /ai-sales-support

COPY . ./

RUN pip install -r requirements.txt

CMD ["python3", "main.py"]
