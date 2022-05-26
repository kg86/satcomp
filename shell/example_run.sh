#!/usr/bin/env bash


scriptpath=`dirname $(readlink -f "$0")` 
cd "$scriptpath"

set -x
if [[ ! -d canterbury-corpus ]]; then
	git clone https://github.com/pfalcon/canterbury-corpus
fi

mkdir -p "$scriptpath"/example_run/log "$scriptpath"/example_run/splitted "$scriptpath"/example_run/datasets
cp canterbury-corpus/canterbury/* "$scriptpath"/example_run/datasets
cp canterbury-corpus/calgary/* "$scriptpath"/example_run/datasets
cp canterbury-corpus/artificial/aaa.txt "$scriptpath"/example_run/datasets
cp canterbury-corpus/artificial/random.txt "$scriptpath"/example_run/datasets
cp canterbury-corpus/large/* "$scriptpath"/example_run/datasets
rm "$scriptpath"/example_run/datasets/SHA1SUM

for dataset in example_run/datasets/*; do 
	[[ -f "$dataset" ]] && [[ -r "$dataset" ]] && "$scriptpath/splitter.sh" $dataset example_run/splitted
done

"$scriptpath/generate_slurmscripts.sh" "$scriptpath/example_run/splitted" "$scriptpath/example_run/log" "$scriptpath/example_run/scripts"
for script in "$scriptpath/example_run/scripts/"*.sh; do
	timeout 1h $script > "$scriptpath/example_run/log/"$(basename $script .sh).log 2> "$scriptpath/example_run/log/"$(basename $script .sh).err
done
"$scriptpath/concat_json.sh" "$scriptpath/example_run/log" > "$scriptpath/example_run/benchmark.json"
