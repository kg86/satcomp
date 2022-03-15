FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git build-essential cmake wget pipenv make curl vim wget pipenv make curl

RUN git clone https://github.com/simongog/sdsl-lite.git \
    && cd sdsl-lite \
    && ./install.sh

COPY . /satcomp
WORKDIR /satcomp

RUN pipenv sync

