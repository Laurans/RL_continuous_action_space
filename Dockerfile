FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory to /workspace
WORKDIR /workspace

RUN mkdir /backup && apt-get update && apt-get install -y git freeglut3-dev x11vnc xvfb wmctrl

COPY requirements_docker.txt ./

# Install any needed packages specified in requirements.txt
RUN pip --no-cache-dir install -r requirements_docker.txt


EXPOSE 5900

COPY vnc.sh /
RUN chmod +x "/vnc.sh"


CMD ["/vnc.sh", "--allow-root"]
