#!/bin/zsh
set -e
set -x

pfad=$HOME/solver/
echo 'LANG=C' > ~/.profile
mkdir -p "$pfad"
cd "$pfad"
tar -xf /solver.tar
sed -i 's@^python_version = "3.8"@python_version = "3.9"@' Pipfile 
pipenv sync 
cd "$pfad/shell"
./example_run.sh 

ln -sv "$pfad/shell/example_run/benchmark.json" "$pfad/shell/docker/scaling.json"
cd "$pfad/shell/docker"
git clone https://github.com/koeppl/sqlplot  
mkdir -p  "$pfad/shell/docker/plot"
$pfad/shell/docker/sqlplot/sqlplot.py -i plot.tex -D scaling.db
./gen_plot.sh
mkdir -p "$pfad/shell/docker/plot"
$pfad/shell/docker/sqlplot/sqlplot.py -i plot_generated.tex
"tar -cf plot.tar plot.tex plot_generated.tex plot/"
