#!/usr/bin/env zsh

function die {
	echo "$1" >&2
	exit 1
}

prefixlength=100
rawfolder="asp_compression_datasets/raw"
dumpfolderbase="asp_compression_datasets/wcnf/$prefixlength/";

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
