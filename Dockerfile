FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git build-essential cmake wget pipenv make curl vim

RUN apt-get install -y wget pipenv make curl

RUN git clone https://github.com/simongog/sdsl-lite.git \
    && cd sdsl-lite \
    && ./install.sh

# RUN apt-get install -y pipenv

COPY . /satcomp
WORKDIR /satcomp


# RUN  git clone https://github.com/pyenv/pyenv.git ~/.pyenv \
#     && export PYENV_ROOT="$HOME/.pyenv" \
#     && export PATH="$PYENV_ROOT/bin:$PATH"    # if `pyenv` is not already on PATH \
#     && eval "$(pyenv init --path)" \
#     && eval "$(pyenv init -)" \
#     && pyenv install 3.9.10


RUN pipenv sync

# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# SHELL ["/bin/bash", "-c"]
# RUN  git clone https://github.com/pyenv/pyenv.git ~/.pyenv \
#     && source .profile \
    # && pyenv install 3.9.10
