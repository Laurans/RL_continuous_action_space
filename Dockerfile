FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory to /workspace
WORKDIR /workspace

RUN apt-get update && apt-get install -y git

COPY requirements_docker.txt ./

# Install any needed packages specified in requirements.txt
RUN pip --no-cache-dir install -r requirements_docker.txt
