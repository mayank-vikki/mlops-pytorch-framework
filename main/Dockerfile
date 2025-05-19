ARG REGION

#If you need GPU support, replace -cpu with -gpu in the image tag (e.g., pytorch-training:2.2.1-gpu-py310...)
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:1.12.1-cpu-py38-ubuntu20.04-sagemaker

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -U \
    flask \
    gevent \
    gunicorn \
    pandas \
    numpy

# Create directories required by SageMaker
RUN mkdir -p /opt/program
RUN mkdir -p /opt/ml

# Copy necessary files
COPY app.py /opt/program
COPY model.py /opt/program
COPY nginx.conf /opt/program
COPY wsgi.py /opt/program
WORKDIR /opt/program

# Expose port 8080 for SageMaker
EXPOSE 8080

# Start the Flask app
ENTRYPOINT ["python", "app.py"]
