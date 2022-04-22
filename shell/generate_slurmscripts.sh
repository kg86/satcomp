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

for filename in $datasetFolder/*; do
	basefilename="$(basename $filename)"
	memory=$kMaxMemory
	putScriptRedir "$Pipenv src/bidirectional_solver.py" "bidir_$basefilename" "$filename" "$memory"
	putScriptRedir "$Pipenv src/attractor_solver.py --algo min" "attr_$basefilename" "$filename" "$memory"
	# putScriptRedir "$Pipenv src/grammar_solver.py" "grammar_$basefilename" "$filename"
	putScriptRedir "$Pipenv src/slp_solver.py" "slp_$basefilename" "$filename" "$memory"
	# putScriptRedir "$Pipenv src/slp_naive.py" "slpnaive_$basefilename" "$filename" "100"
done 
