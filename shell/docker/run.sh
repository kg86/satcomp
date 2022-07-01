#!/bin/zsh
set -e
set -x

function die {
	echo "$1" >&2
	exit 1
}

pfad=$HOME/solver/
mkdir -p "$pfad"
cd "$pfad"
tar -xf /solver.tar
sed -i 's@^python_version = "3.8"@python_version = "3.9"@' Pipfile 
pipenv sync 
cd "$pfad/shell"
[[ -d "$pfad/shell/example_run" ]] && rm -r "$pfad/shell/example_run" 
./example_run.sh 
[[ -r "$pfad/shell/example_run/benchmark.json" ]] || die "cannot read benchmark.json - something went wrong!"

[[ -e "$pfad/shell/docker/scaling.json" ]] && rm "$pfad/shell/docker/scaling.json"
[[ -h "$pfad/shell/docker/scaling.json" ]] && rm "$pfad/shell/docker/scaling.json"
ln -sv "$pfad/shell/example_run/benchmark.json" "$pfad/shell/docker/scaling.json"
cd "$pfad/shell/docker"
[[ -d sqlplot ]]  || git clone https://github.com/koeppl/sqlplot  
mkdir -p  "$pfad/shell/docker/plot"
[[ -e "scaling.db" ]] && rm scaling.db
$pfad/shell/docker/sqlplot/sqlplot.py -i "$pfad/shell/docker/plot.tex" -D scaling.db
./gen_plot.sh
mkdir -p "$pfad/shell/docker/plot"
$pfad/shell/docker/sqlplot/sqlplot.py -i "$pfad/shell/docker/plot_generated.tex"
tar -cf plot.tar plot.tex plot_generated.tex plot/
