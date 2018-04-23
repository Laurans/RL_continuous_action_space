# Reinforcement learning in continuous space

Workspace to design an agent that works in continuous state and action space with actor-critic methods.
Currently, I am developing an implementation of **Deep Deterministic Policy Gradients** on the *MountainCarContinuous* environment.

Original paper:
> Lillicrap, Timothy P., et al., 2015. **Continuous Control with Deep Reinforcement Learning**. [[pdf](https://arxiv.org/pdf/1509.02971.pdf)]

## Installation

### With Docker

#### Requirements
* Docker 18.03 and up
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 2.0 and up

#### Building docker image
```
git clone https://github.com/Laurans/RL_continuous_action_space.git
cd RL_continuous_action_space
docker build -t rl_continuous_space .
```
#### Running image
```nvidia-docker run -it -v `pwd`:/workspace/ --name rlcontinuous rl_continuous_space bash```

### With Pip
#### Requirements
* Tensorflow GPU
* Keras 2.0 and up
* Python 3.5 and up

#### Installation
`pip install -r requirements.txt` 

## Usage

```python
python main.py
```

## Development

Look at the `config.yml` file to change the parameters of the algorithm.

### Unit testing usage
```
python -m unittest tests/<script>.py -v
```

## Thanks

VNC configuration for docker was inspired by this repo [pascalwhoop/tf_openailab_gpu_docker](https://github.com/pascalwhoop/tf_openailab_gpu_docker)
Sum tree inspired by takoika/PrioritizedExperienceReplay

## License
[MIT](https://choosealicense.com/licenses/mit/)
