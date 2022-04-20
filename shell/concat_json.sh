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
	filebasename=$(basename $filename)
	algo=$(echo $filebasename | cut -f1 -d'_')
	dataset=$(echo $filebasename | cut -f2- -d'_')
	datasetbasename="$(basename $dataset .log)"
	if echo "$datasetbasename" | grep -q '\.[0-9][0-9][0-9][0-9]$'; then
		file_len=$(echo "$datasetbasename" | sed 's@.*\.\([0-9][0-9][0-9][0-9]\)$@\1@')
		datasetbasename=$(echo "$datasetbasename" | sed 's@\.[0-9][0-9][0-9][0-9]$@@')
	fi
	# echo $dataset
	# return
	
	if ! grep -q '^{' "$filename"; then
		[[ "$algo" = "attr" ]] && continue
		errfile=$filename:r.err
		Status='unknown'
		grep -q "DUE TO TIME LIMIT" "$errfile" && Status="no time"
		grep -q "cgroup out-of-memory" "$errfile" && Status="no mem"
		[[ "$counter" -gt 0 ]] && echo ', '
		((counter++))
		echo "{\"algo\": \"$algo\", \"status\": \"$Status\", \"file_name\": \"$dataset\", \"dataset\": \"$datasetbasename\", \"file_len\": \"$file_len\" }"
		continue
	fi
	[[ "$counter" -gt 0 ]] && echo ', '
	((counter++))
	grep '^{' "$filename" | sed "s@\"algo\": \"[^\"]\+\"@\"algo\": \"$algo\",\"dataset\": \"$datasetbasename\"@"
done
echo ']'

