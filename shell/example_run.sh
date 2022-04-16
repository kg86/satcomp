#!/bin/bash

set -x

[[ -d canterbury-corpus ]] || git clone https://github.com/pfalcon/canterbury-corpus
mkdir -p log
./generate_slurmscripts.sh canterbury-corpus/canterbury log
for script in ./slurmscripts/*.sh; do
	$script
done
./concat_json.sh log > benchmark.json

for dataset in ~/data/calgary/*; do ./splitter.sh $dataset ~/data/splitted; done
