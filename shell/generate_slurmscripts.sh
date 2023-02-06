#!/usr/bin/env bash

kMaxMemory='16000'

function die {
	echo "$1" >&2
	exit 1
}

[[ "$#" -eq 3 ]] || 
	die "Usage: $0 [dataset-folder] [log-folder] [script-folder]
	- dataset-folder: folder in which the datasets are stored
	- log-folder: folder in which the log files to write
	- script-folder: folder in which the executable scripts to write"

[[ -d "$1" ]] || die "$1 is not a directory"
datasetFolder=$(readlink -e "$1")
[[ -d "$datasetFolder" ]] || die "$datasetFolder is not a directory"

mkdir -p "$2" || die "cannot create folder $2"
logFolder=$(readlink -e "$2")
[[ -d "$logFolder" ]] || die "$logFolder is not a directory"

mkdir -p "$3" || die "cannot create folder $3"
scriptFolder=$(readlink -e "$3")
[[ -d "$scriptFolder" ]] || die "$scriptFolder is not a directory"


scriptpath=`dirname $(readlink -f "$0")`

function putScript {
command="$1"
basename="$2"
filename="$3"
memory="$4"
cat <<EOF
#!/bin/zsh
#
#SBATCH --job-name=${basename} # Job name
#SBATCH --ntasks=9                    # Run on a single CPU
#SBATCH --mem=${memory}                     # memory in megabyte
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --output=$logFolder/${basename}.log   # stdout
#SBATCH --error=$logFolder/${basename}.err    # stderr
#
#PBS -N ${basename}
#PBS -l walltime=01:00:00,mem=${memory}mb,nodes=1,procs=1
#PBS -o $logFolder/${basename}.log
#PBS -e $logFolder/${basename}.err



cd $scriptpath/../
if [[ -n "\$SLURM_JOB_ID" ]] || [[ -n "\$PBS_JOBID" ]]; then
	$command --file "$filename"
else
	$command --file "$filename" >  "$logFolder/${basename}.log" 2> "$logFolder/${basename}.err"
fi
EOF
}


function putScriptRedir {
	command="$1"
	basename="$2"
	filename="$3"
	memory="$4"
	putScript "$command" "$basename" "$filename" "$memory" > "$scriptFolder/${basename}.sh"
	chmod u+x "$scriptFolder/${basename}.sh"
}


Pipenv="pipenv run python"
function prepare_generator {
	for solver in Glucose4 Cadical; do
		for strategy in RC2 LSU FM; do
			[[ $solver == Cadical ]] && [[ $strategy == LSU ]] && continue
			prefixlengthname=$(printf "%07.f" "$prefixlength")
			basefilename="${compressionmeasure}_${strategy}_${solver}_$(basename $filename)_${prefixlengthname}"
			memory=$kMaxMemory
			putScriptRedir "$Pipenv src/${compressionmeasure}_solver.py --prefix "$prefixlength" --solver "$solver" --strategy $strategy" "${basefilename}" "$filename" "$memory"
		done 
	done
}

for filename in $datasetFolder/*; do
	filesize=$(stat --printf="%s" "$filename")
	for prefixlength in $(seq 100 20 300); do
		if [[ $prefixlength -gt $filesize ]]; then
			break
		fi
		for compressionmeasure in bms attractor slp; do
				prepare_generator
		done
	done
done

for filename in $datasetFolder/*; do
	filesize=$(stat --printf="%s" "$filename")
	for prefixlength in $(seq 100000 100000 2000000); do
		if [[ $prefixlength -gt $filesize ]]; then
			break
		fi
		for compressionmeasure in attractor; do
				prepare_generator
		done
	done
done

