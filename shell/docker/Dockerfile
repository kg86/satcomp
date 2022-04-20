# vim: set syntax=dockerfile:

FROM debian:stable
ARG DEBIAN_FRONTEND=noninteractive
ARG PACKAGE_DEPENDENCY="\
python3-pip pipenv \
zsh git vim sqlite3  \
"


# slurm-wlm cgroup-tools slurm-client 
# COPY cgroup.conf /etc/slurm-llnl/
# COPY slurm.conf /etc/slurm-llnl

RUN \
apt-get update && \
apt-get -qy dist-upgrade && \
apt-get install -qy  && \
# locales \
apt-get install -y  $PACKAGE_DEPENDENCY && \
apt-get install -f && \
mkdir -p /home/share && \
groupadd -g 1001 share && \
useradd -u 1001 -g 1001 share -G users,uucp,games,cdrom,floppy,audio,dip,video,plugdev && \
chown share /home/share && \
chsh -s /bin/bash share && \
echo 'root:a\nshare:a' | chpasswd  && \
echo 'LANG=C' > /home/share/.profile
# cgcreate -g memory:myGroup && \
# echo 16000M > /sys/fs/cgroup/memory/myGroup/memory.limit_in_bytes
COPY --chown=share run.sh /
COPY --chown=share solver.tar /

ENV HOME /home/share
WORKDIR /home/share
USER share

