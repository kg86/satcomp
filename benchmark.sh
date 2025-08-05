#!/bin/sh
set -x
set -e

die() {
	echo "$1" >&2
	exit 1
}
scriptpath=`dirname $(readlink -f "$0")`


# download calgary and canterbury datasets
if [ -z "$2" ]; then
	die "Usage: $0 <dataset_dir> <log_dir>"
fi
datasetFolder="$(readlink -f "$1")"
wholeFolder="$datasetFolder/whole"
logFolder="$(readlink -f "$2")"
mkdir -p "$logFolder" || die "Failed to create directory $logFolder"

if [ ! -d "$datasetFolder" ]; then
	mkdir -p "$datasetFolder" || die "Failed to create directory $datasetFolder"
fi

#check whether $datasetFolder is nonempty
if [ ! -d "$datasetFolder/whole" ]; then
	cd $datasetFolder || die "Failed to change directory to $datasetFolder"

	if [ ! -d "$datasetFolder/artificial_datasets" ]; then
		git clone https://github.com/koeppl/artificial_datasets || die "Failed to clone canterbury-corpus"
	fi
	cp "$datasetFolder/artificial_datasets"/* $wholeFolder/ || die "Failed to copy artificial_datasets files to $wholeFolder directory"

	if [ ! -d "$datasetFolder/canterbury-corpus" ]; then
		git clone https://github.com/pfalcon/canterbury-corpus || die "Failed to clone canterbury-corpus"
	fi
	mkdir whole || die "Failed to create directory whole"

	cd canterbury-corpus
	cp canterbury/* $wholeFolder/ || die "Failed to copy canterbury-corpus files to $wholeFolder directory"
	cp calgary/* $wholeFolder/ || die "Failed to copy calgary-corpus files to $wholeFolder directory"
	cp large/* $wholeFolder/ || die "Failed to copy large-corpus files to $wholeFolder directory"
	cp artificial/* $wholeFolder/ || die "Failed to copy artificial-corpus files to $wholeFolder directory"


	rm $wholeFolder/SHA1SUM
fi

cd "$scriptpath" || die "Failed to change directory to $scriptpath"
for file in "$wholeFolder"/*; do 
	 pipenv run python3 tests/benchmark.py --file "$file" --outdir "$logFolder" --maxtime 3600 --maxmem 64000000000; 
 done
