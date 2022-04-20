#!/bin/bash
set -x
set -e
tmpfile=$(mktemp)
tar -cf $tmpfile ../../
mv $tmpfile solver.tar
docker build --memory 16G -t satsolver .
docker run -it --memory=16G sat 
