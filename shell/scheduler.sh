#!/bin/zsh

set -e

local -r memory_pattern="^MemTotal: *\([0-9]\+\) *kB"
local -r maxmem_pattern='^#SBATCH --mem=\([0-9]\+\).*'

total_memory=$(grep "$memory_pattern" /proc/meminfo | sed "s@$memory_pattern@\1@")
total_allowed_memory=$(calc "round(${total_memory} * 0.8)")

#TODO: add array of processes, maintain maximum_memory -> schedule

myname=$(whoami)

for i in $(seq -f "%03.f" $(nproc)); do 
	echo $i
	sudo cgcreate -a "$myname" -t "$myname" -g memory:myexp${i}
	sudo chmod a+rx  /sys/fs/cgroup/myexp${i}
	echo "0" | tee /sys/fs/cgroup/myexp${i}/memory.max
	echo '0' | tee /sys/fs/cgroup/myexp${i}/memory.swap.max
	echo "+cpu" | tee /sys/fs/cgroup/myexp${i}/cgroup.subtree_control
done


function maintenance {
	for i in $(seq -f "%03.f" $(nproc)); do 
		if [[ $(cat /sys/fs/cgroup/myexp${i}/cgroup.procs | wc -l) -eq 0 ]]; then
			echo "0" | tee /sys/fs/cgroup/myexp${i}/memory.max >/dev/null
			export freeprocess="$i"
		fi
	done
	total_reserved_memory=$(cat /sys/fs/cgroup/myexp*/memory.max | tr '\n' '+' )
	total_reserved_memory=$(echo "$total_reserved_memory 0" | bc)
}

for scriptfile in calgary/*.sh; do

	freeprocess=''
	while [[ -z $freeprocess ]]; do
		maintenance
		if [[ -z $freeprocess ]]; then 
			echo "no free process ..."
			sleep 1
		fi
	done

	desired_memory=$(cat "$scriptfile" | grep "$maxmem_pattern" | sed "s@$maxmem_pattern@\1@")
	desired_memory="${desired_memory}000"

	while [[ $(expr $desired_memory + $total_reserved_memory) -gt $total_allowed_memory ]];  do
		echo "no free memory... used: $total_reserved_memory needed: $desired_memory limit: $total_allowed_memory"
		maintenance
		sleep 1
	done

	echo "${desired_memory}" | tee /sys/fs/cgroup/myexp${freeprocess}/memory.max

	./$scriptfile &
	pid=$!
	echo "$pid" | sudo tee /sys/fs/cgroup/myexp${freeprocess}/cgroup.procs

	# cgexec -g memory:myexp${freeprocess} "$scriptfile" &
done
wait

# sudo cgexec -g memory:myexp zsh ./calgary/attractor_FM_aaa.txt_0010.sh
echo "finished"
