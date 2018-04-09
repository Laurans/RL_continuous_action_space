FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory to /workspace
WORKDIR /workspace

RUN apt-get update && apt-get install git && git clone https://github.com/openai/gym.git && cd gym && pip install -e .

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run jupyter when container launches
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
