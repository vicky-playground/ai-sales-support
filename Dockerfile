FROM python:3.10

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

WORKDIR /ai-sales-support

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "/ai-sales-support/main.py"]
