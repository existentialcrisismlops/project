# Use a base image with Python
FROM python:3.9-slim

# Install Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add Conda to the system PATH
ENV PATH /opt/conda/bin:$PATH

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the Conda environment specification
COPY conda.yaml .

# Update Conda and create the environment
RUN conda update -n base -c defaults conda \
    && conda env create -f conda.yaml \
    && echo "conda activate predictive_maintenance" >> ~/.bashrc

# Copy the rest of the application code
COPY app/ .

# Expose the Flask application port
EXPOSE 5000

# Set the entrypoint command to run the Flask application
CMD ["python", "app.py"]
