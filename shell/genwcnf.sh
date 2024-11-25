#!/usr/bin/env zsh

function die {
	echo "$1" >&2
	exit 1
}

rawfolder="asp_compression_datasets/raw"
dumpfolderbase="asp_compression_datasets/wcnf/$prefixlength/";

[[ $# -ne 1 ]] && 
	die "Usage: $0 prefixsize\nAssume that $rawfolder contains all input datasets\nWrites to folder $dumpfolderbase"

[[ -d "$rawfolder" ]] || die "Folder $rawfolder should contain all input datasets"


dumpfolderbase="asp_compression_datasets/wcnf/$prefixlength/";
prefixlength="$1"

echo "using prefix length: $prefixsize"
echo "input folder: $rawfolder"
echo "output folder: $dumpfolderbase"


set -x
set -e
for f in "$rawfolder"/*; do 
	b=$(basename "$f"); 
	for sol in attractor_solver.py slp_solver.py bms_solver.py slp_fast.py; do
		prg="src/$sol"
		[[ -r $prg ]] || die "cannot evaluate $prg!"

		solbase=$(basename "$prg" ".py")
		dumpfolder="$dumpfolderbase/$solbase"
		mkdir -p "$dumpfolder"
		pipenv run python "$prg" --file "$f" --prefix "$prefixlength" --dump "$dumpfolder/$b"
	done
done
