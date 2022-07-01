#!/usr/bin/env bash

function die {
	echo "$1" >&2
	exit 1
}

[[ "$#" -eq 2 ]] || 
	die "Usage: $0 [dataset-folder] [log-folder]
	- dataset-folder: folder in which the datasets are stored
	- log-folder: folder in which the log files to write
"


[[ -d "$1" ]] || die "$1 is not a directory"
datasetFolder=$(readlink -e "$1")
[[ -d "$datasetFolder" ]] || die "$datasetFolder is not a directory"

mkdir -p "$2" || die "cannot create folder $2"
logFolder=$(readlink -e "$2")
[[ -d "$logFolder" ]] || die "$logFolder is not a directory"

set -e
set -x


scriptpath=`dirname $(readlink -f "$0")` 

oldpwd=$(pwd)
cd $scriptpath/..
git submodule update --init --recursive
mkdir -p $scriptpath/../externals/lzrr/build
cd $scriptpath/../externals/lzrr/build
cmake ..
make
cd "$oldpwd"

outfile=$(mktemp)


for file in $datasetFolder/*; do
	basefile=$(basename "$file")
	for method in lzrr lz lex; do
		/usr/bin/time -f "RESULT method=${method} file=${basefile}, timesec=%e, memorykb=%M" $scriptpath/../externals/lzrr/build/compress.out -i ${file} -o "$outfile" -m ${method} 2>&1 | tee $logFolder/${method}_${basefile}.log 
	done
done
rm "$outfile"
