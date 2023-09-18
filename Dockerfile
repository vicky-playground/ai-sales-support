FROM python:3.10

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

WORKDIR /ai-sales-support

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
