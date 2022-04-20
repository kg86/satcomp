#!/bin/bash
scriptpath=`dirname $(readlink -f "$0")` 
cd "$scriptpath"

set -x
set -e
tmpfile=$(mktemp)
tar -cf $tmpfile ../../
mv $tmpfile solver.tar
docker build --memory 16G -t satsolver .
container_id=$(docker run --memory=16G -d -it --name satsolver satsolver)
docker exec -it $container_id bash /run.sh
docker cp $container_id:/home/share/solver/shell/docker/plot.tar .
docker container stop $container_id 
docker container rm $container_id 
