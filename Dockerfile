FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git build-essential cmake wget pipenv make curl vim wget pipenv make curl

# install sdsl
RUN git clone https://github.com/simongog/sdsl-lite.git \
    && cd sdsl-lite \
    && ./install.sh

COPY . /satcomp
WORKDIR /satcomp

# install lzrr
RUN git clone https://github.com/TNishimoto/lzrr.git externals/lzrr \
    && cd externals/lzrr \
    && git submodule init \
    && git submodule update \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make

RUN pipenv sync


