# Defaults to the CPU version
ARG tensorflow_version=latest 
ARG time_zone=Etc/UTC

FROM tensorflow/tensorflow:${tensorflow_version} AS experiment_runner
WORKDIR /
COPY pythonPackages.txt /
RUN pip install --upgrade pip
RUN pip install -r pythonPackages.txt

# Adding sudo access
RUN apt-get update \
 && apt-get install -y sudo

RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


# When we want all dependancies in the image
FROM experiment_runner AS experiment_complete

ENV TZ='${time_zone}'
RUN apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/${time_zone} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Install terraform
USER docker
RUN sudo apt-get update
RUN sudo apt-get install -y gnupg software-properties-common curl
RUN curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
RUN sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
RUN sudo apt-get update && sudo apt-get install terraform

# Only setting up experiment
FROM hashicorp/terraform:latest as experiment_setup_only
WORKDIR /AWS_IaC
# TODO Add test runner dependancies

