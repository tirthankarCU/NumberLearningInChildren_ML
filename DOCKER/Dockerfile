# getting docker base image
FROM dokken/ubuntu-20.04

# Update package repository and install required dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3.10

# Install Git
RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/tirthankarCU/NLP_RL_Docker_Version.git

WORKDIR ~/NLP_RL_Docker_Version