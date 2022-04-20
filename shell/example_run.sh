#!/bin/bash
set -x
if [[ ! -d canterbury-corpus ]]; then
	git clone https://github.com/pfalcon/canterbury-corpus
fi

mkdir -p example_run/log example_run/splitted example_run/datasets
cp canterbury-corpus/canterbury/* example_run/datasets
cp canterbury-corpus/calgary/* example_run/datasets
cp canterbury-corpus/artificial/aaa.txt example_run/datasets
cp canterbury-corpus/artificial/random.txt example_run/datasets
cp canterbury-corpus/large/* example_run/datasets
rm example_run/datasets/SHA1SUM

for dataset in example_run/datasets/*; do 
	[[ -f "$dataset" ]] && [[ -r "$dataset" ]] && ./splitter.sh $dataset example_run/splitted
done

./generate_slurmscripts.sh example_run/splitted example_run/log example_run/scripts
for script in example_run/scripts/*.sh; do
	timeout 1h $script > example_run/log/$(basename $script .sh).log 2> example_run/log/$(basename $script .sh).err
done
./concat_json.sh example_run/log > example_run/benchmark.json
