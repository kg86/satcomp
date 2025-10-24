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

if [ ! -d "$datasetFolder/sqlplot" ]; then
	cd $datasetFolder || die "Failed to change directory to $datasetFolder"
	git clone "https://github.com/koeppl/sqlplot" || die "Failed to clone sqlplot"
fi

#check whether $datasetFolder is nonempty
if [ ! -d "$datasetFolder/whole" ]; then
	cd $datasetFolder || die "Failed to change directory to $datasetFolder"
	mkdir whole || die "Failed to create directory whole"

	if [ ! -d "$datasetFolder/artificial_datasets" ]; then
		git clone https://github.com/koeppl/artificial_datasets || die "Failed to clone canterbury-corpus"
	fi

	cp "$datasetFolder/artificial_datasets"/* $wholeFolder/ || die "Failed to copy artificial_datasets files to $wholeFolder directory"

	if [ ! -d "$datasetFolder/canterbury-corpus" ]; then
		git clone https://github.com/pfalcon/canterbury-corpus || die "Failed to clone canterbury-corpus"
	fi

	cd canterbury-corpus
	cp canterbury/* $wholeFolder/ || die "Failed to copy canterbury-corpus files to $wholeFolder directory"
	cp calgary/* $wholeFolder/ || die "Failed to copy calgary-corpus files to $wholeFolder directory"
	cp large/* $wholeFolder/ || die "Failed to copy large-corpus files to $wholeFolder directory"
	cp artificial/* $wholeFolder/ || die "Failed to copy artificial-corpus files to $wholeFolder directory"


	rm $wholeFolder/SHA1SUM
fi

cd "$scriptpath/../.." || die "Failed to change directory to $scriptpath/../.."


# Experiment for Figure 6

if [ ! -d "$logFolder/fig6" ]; then
	mkdir -p "$logFolder/fig6" || die "Failed to create directory $logFolder"
fi

MINPREFIX=200
MAXPREFIX=600

# MINPREFIX=5
# MAXPREFIX=10

pipenv run python3 "$scriptpath/fig_benchmark.py" --file "$wholeFolder/fibonacci.20" --maxtime 3600 --maxmem 16000000000 --maxprefix ${MAXPREFIX} --minprefix ${MINPREFIX} --step 100 --outdir "$logFolder/fig6" --fig 6; 
pipenv run python3 "$scriptpath/fig_benchmark.py" --file "$wholeFolder/news" --maxtime 3600 --maxmem 16000000000 --maxprefix ${MAXPREFIX} --minprefix ${MINPREFIX} --step 100 --outdir "$logFolder/fig6" --fig 6; 
pipenv run python3 "$scriptpath/fig_benchmark.py" --file "$wholeFolder/paper1" --maxtime 3600 --maxmem 16000000000 --maxprefix ${MAXPREFIX} --minprefix ${MINPREFIX} --step 100 --outdir "$logFolder/fig6" --fig 6; 
pipenv run python3 "$scriptpath/fig_benchmark.py" --file "$wholeFolder/asyoulik.txt" --maxtime 3600 --maxmem 16000000000 --maxprefix ${MAXPREFIX} --minprefix ${MINPREFIX} --step 100 --outdir "$logFolder/fig6" --fig 6; 

cd "$scriptpath"
jsons=("$logFolder/fig6"/*.json)
if ((${#jsons[@]})); then
  grep -l 'error' "${jsons[@]}" | xargs -r rm -f -- || true
fi
python3 "$scriptpath/concat_json.py" "$logFolder/fig6/"*.json > "$scriptpath/fig6.json"
$datasetFolder/sqlplot/sqlplot.py -i "$scriptpath/fig6.tex"
latexmk -pdf "$scriptpath/fig6.tex"
