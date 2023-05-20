# getting docker base image
FROM dokken/ubuntu-20.04

# Update package repository and install required dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3.10.11

# Set the default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10.11

# Verify the Python version
RUN python --version

# Install Git
RUN apt-get update && apt-get install -y git

RUN git clone git@github.com:tirthankarCU/NLP_RL_Docker_Version.git ~/