#!/bin/bash
set -e
set -x

scriptpath=`dirname $(readlink -f "$0")`

function putScript {
command="$1"
basename="$2"
filename="$3"
cat <<EOF
#!/bin/zsh
#
#SBATCH --job-name=${basename} # Job name
#SBATCH --ntasks=9                    # Run on a single CPU
#SBATCH --mem=16000                     # memory in megabyte
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --output=${HOME}/log/${basename}.log   # stdout
#SBATCH --error=${HOME}/log/${basename}.err    # stderr

cd $scriptpath/../
$command --file "$filename"
EOF
}

mkdir -p slurmscripts

function putScriptRedir {
	command="$1"
	basename="$2"
	putScript "$command" "$basename" "$filename" > "slurmscripts/${basename}.sh"
}


Pipenv="pipenv run python"

for filename in $HOME/data/*; do
	basefilename="$(basename $filename)"
	putScriptRedir "$Pipenv src/bidirectional_solver.py" "bidir_$basefilename" "$filename"
	putScriptRedir "$Pipenv src/attractor_solver.py --algo min" "attr_$basefilename" "$filename"
	putScriptRedir "$Pipenv src/grammar_solver.py" "grammar_$basefilename" "$filename"
	putScriptRedir "$Pipenv src/slp_solver.py" "slp_$basefilename" "$filename"
done 
