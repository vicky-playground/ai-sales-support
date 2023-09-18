FROM python:3.9
COPY . /ai-sales-support
WORKDIR /ai-sales-support
RUN pip install -r requirements.txt
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
ENTRYPOINT ["python3", "main.py"]
