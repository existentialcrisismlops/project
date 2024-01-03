FROM python:3.11.3
WORKDIR /proj
COPY requirements.txt /proj/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /proj
EXPOSE 8080
CMD ["python", "app.py"]