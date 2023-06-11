FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN apt-get update
RUN apt-get install -y vim

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

CMD ["python", "train.py"]
