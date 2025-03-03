FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir transformers==4.37.0
RUN pip install --no-cache-dir accelerate

RUN pip install --no-cache-dir gunicorn

COPY main.py main.py

EXPOSE 7982

CMD ["gunicorn", "--bind", "0.0.0.0:7982", "-w", "8", "main:app"]
# gunicorn --bind 0.0.0.0:7982 -w 2 main:app

# docker build -t reedless/aycep:finals .
# docker push reedless/aycep:finals

# curl --location 'https://79hydobncljbip-7982.proxy.runpod.net/query' --header 'Content-Type: application/json' --data '{"input": "your_input"}'
