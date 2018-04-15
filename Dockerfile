FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory to /workspace
WORKDIR /workspace

RUN apt-get update && apt-get install -y git freeglut3-dev x11vnc xvfb fluxbox wmctrl

COPY requirements_docker.txt ./

# Install any needed packages specified in requirements.txt
RUN pip --no-cache-dir install -r requirements_docker.txt


EXPOSE 5900

# Make port 8888 available to the world outside this container
EXPOSE 8888

COPY vnc.sh /
RUN chmod +x "/vnc.sh"


CMD ["/vnc.sh", "--allow-root"]
