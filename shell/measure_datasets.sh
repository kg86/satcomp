#!/bin/zsh
scriptpath=`dirname $(readlink -f "$0")` 
cd "$scriptpath"
mkdir -p measure/log measure/scripts
for datafolder in $scriptpath/../data/*; do
	[[ -d "$datafolder" ]] || continue
	./generate_slurmscripts.sh "$datafolder" measure/log measure/scripts
done

if which sbatch >/dev/null 2>&1; then
	for i in measure/scripts/*.sh; do 
		sbatch $i; 
	done
else
	for script in measure/scripts/*.sh; do 
		timeout 1h $script > measure/log/$(basename $script .sh).log 2> measure/log/$(basename $script .sh).err
	done
	./concat_json.sh measure/log > measure/benchmark.json
	cat measure/benchmark.json | jq '[.[] | {algo: .["algo"], file: .["file_name"], outputsize: .["factor_size"] }]' | tee measure/stats.json
fi
