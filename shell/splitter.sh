#!/usr/bin/env zsh

function die {
	echo "$1" >&2
	exit 1
}

[[ "$#" -eq 2 ]] || die "Usage: $0 [dataset] [output-folder]"
dataset="$1"
datasetFolder=$(readlink -e "$2")
[[ -r "$dataset" ]] || die "$dataset not readable"
[[ -d "$2" ]] || die "$2 is not a directory"
[[ -d "$datasetFolder" ]] || die "$datasetFolder is not a directory"

base=$(basename "$dataset")
# for i in $(seq -f "%04.f" 2 3 7); do
for i in $(seq -f "%04.f" 10 10 200) $(seq -f "%04.f" 200 50 800) $(seq -f "%04.f" 800 200 3000); do
	dd if="$dataset" of="$datasetFolder/$base.$i" bs=$i count=1
done
