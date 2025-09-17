FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
# this means all packages install at one go

EXPOSE 5000
# give port of flask because we build flask application

CMD ["python","app.py"]
# this is a command to run flask application
