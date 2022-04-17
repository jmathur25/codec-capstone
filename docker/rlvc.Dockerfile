# Build an image for us to run RLVC. We need this cause RLVC uses an old version of tensorflow (1.2.0)
# which requires an old version of CUDA (9.0). The host has CUDA version 11.6. So, we use a
# Docker container and run the 9.0 driver in the container. This avoids conflicts on the host.
# A simple manual setup is shown below. However, a container should already be setup and running
# with volume mounts for ease of development.
# Build:
# sudo docker build . -f rlvc.Dockerfile -t rlvc
# Run:
# sudo docker run --gpus all -it --rm rlvc bash
# (in the container)
# python3
# import tensorflow as tf
# tf.test.is_gpu_available() # returns True

FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get update
# SSH things
RUN apt-get install -y openssh-server sudo
RUN mkdir -p ~/.ssh
RUN touch ~/.ssh/authorized_keys
RUN service ssh restart

# Make pip up to date
RUN python -m pip install --upgrade pip

# Build bpg binaries (bpgenc, bpgdec)
RUN wget https://bellard.org/bpg/libbpg-0.9.8.tar.gz
RUN tar -xzvf libbpg-0.9.8.tar.gz
RUN sudo apt-get install -y cmake libsdl2-dev yasm libpng16-dev
RUN cd libbpg-0.9.8
RUN make -j
RUN sudo make install

# SSH port
EXPOSE 22

CMD ["sleep", "infinity"]
