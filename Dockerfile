FROM apache/airflow:2.9.0

USER root
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt