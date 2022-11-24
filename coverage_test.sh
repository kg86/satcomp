#!/bin/zsh
local -r kTempDir="$(mktemp -d)"
trap 'rm -rf -- "$kTempDir"' EXIT
file="$kTempDir/fib06"
echo 'abaababaabaababaababa' > "$file"
outfile="$kTempDir/outfile"

set -e
set -x
for strategy in RC2 LSU FM; do
	pipenv run python src/slp_solver.py --file "$file"       --strategy "$strategy" --output "$outfile" &&
		pipenv run python src/slp_decoder.py --json "$outfile"
	pipenv run python src/bms_solver.py --file "$file"       --strategy "$strategy" --output "$outfile" &&
		pipenv run python src/bms_decoder.py --json "$outfile"

	pipenv run python src/slp_solver.py --file "$file"       --strategy "$strategy" | pipenv run python src/slp_verify.py --file "$file"
	pipenv run python src/bms_solver.py --file "$file"       --strategy "$strategy" | pipenv run python src/bms_verify.py --file "$file"
	pipenv run python src/attractor_solver.py --file "$file" --strategy "$strategy" | pipenv run python src/attractor_verify.py --file "$file"

done
for type in exact atmost; do
	pipenv run python src/attractor_solver.py --file "$file" --algo "$type" --size 4 | pipenv run python src/attractor_verify.py --file "$file"
done
