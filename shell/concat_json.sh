#!/bin/zsh

function die {
	echo "$1" >&2
	exit 1
}

[[ "$#" -eq 1 ]] || die "Usage: $0 [log-folder]"
logFolder=$(readlink -e "$1")
[[ -d "$logFolder" ]] || die "$logFolder is not a directory"

echo '['
((counter=0))
for filename in $logFolder/*.log; do
	[[ "$counter" -gt 0 ]] && echo ', '
	((counter++))
	algo=$(echo $filename | cut -f1 -d'_')
	dataset=$(echo $filename | cut -f2- -d'_')
	if ! grep -q '^{' "$filename"; then
		errfile=$filename:r.err
		grep -q "DUE TO TIME LIMIT" "$errfile" && Status="no time"
		grep -q "cgroup out-of-memory" "$errfile" && Status="no mem"
		echo "{\"algo\": \"$algo\", \"status\": \"$Status\", \"file_name\": \"$dataset\" }"
		continue
	fi
	grep '^{' "$filename" | sed "s@\"solver\"@\"$algo\"@"
done
echo ']'

