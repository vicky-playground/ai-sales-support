FROM python:3.10
COPY . /ai-sales-support
WORKDIR /ai-sales-support

RUN pip install -r requirements.txt

ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

CMD ["python", "main.py"]
