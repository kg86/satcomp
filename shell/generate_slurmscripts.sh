#!/bin/bash
# set -e
# set -x
kMaxMemory='16000'

function die {
	echo "$1" >&2
	exit 1
}

[[ "$#" -eq 2 ]] || die "Usage: $0 [dataset-folder] [log-folder]"
datasetFolder=$(readlink -e "$1")
logFolder=$(readlink -e "$2")
[[ -d "$1" ]] || die "$1 is not a directory"
[[ -d "$2" ]] || die "$2 is not a directory"
[[ -d "$datasetFolder" ]] || die "$datasetFolder is not a directory"
[[ -d "$logFolder" ]] || die "$logFolder is not a directory"


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

cd $scriptpath/../
if [[ -n "\$SLURM_JOB_ID" ]]; then
	$command --file "$filename"
else
	$command --file "$filename" >  "$logFolder/${basename}.log" 2> "$logFolder/${basename}.err"
fi
EOF
}

mkdir -p slurmscripts

function putScriptRedir {
	command="$1"
	basename="$2"
	filename="$3"
	memory="$4"
	putScript "$command" "$basename" "$filename" "$memory" > "slurmscripts/${basename}.sh"
	chmod u+x "slurmscripts/${basename}.sh"
}


Pipenv="pipenv run python"

for filename in $datasetFolder/*; do
	basefilename="$(basename $filename)"
	memory=$kMaxMemory
	putScriptRedir "$Pipenv src/bidirectional_solver.py" "bidir_$basefilename" "$filename" "$memory"
	putScriptRedir "$Pipenv src/attractor_solver.py --algo min" "attr_$basefilename" "$filename" "$memory"
	# putScriptRedir "$Pipenv src/grammar_solver.py" "grammar_$basefilename" "$filename"
	putScriptRedir "$Pipenv src/slp_solver.py" "slp_$basefilename" "$filename" "$memory"
	putScriptRedir "$Pipenv src/slp_naive.py" "slpnaive_$basefilename" "$filename" "100"
done 
