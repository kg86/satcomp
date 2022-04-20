#!/usr/bin/env zsh


datasets=(
aaa.txt
E.coli
alice29.txt
alphabet.txt
asyoulik.txt
bib
bible.txt
book1
book2
cp.html
fields.c
geo
grammar.lsp
kennedy.xls
lcet10.txt
news
obj1
obj2
pi.txt
pic
plrabn12.txt
progc
progl
progp
ptt5
random.txt
sum
trans
world192.txt
xargs.1)
if [[ -f "scaling.db" ]]; then
    datasets=()
    for i in $(echo "select dataset from scaling group by dataset;" | sqlite3 scaling.db); do
	datasets+="$i"
    done
fi

attributes='xmode=log,ymode=log'


function genPlot {

cat<<"EOF"

%% IMPORT-JSON-DATA scaling scaling.json

%% DEFINE myplot(column,filename)
%% SELECT 
%% file_len AS x, 
%% $column as y,
%% MULTIPLOT FROM scaling
%% WHERE dataset = $filename
%% AND $column IS NOT NULL
%% GROUP BY MULTIPLOT,x ORDER BY MULTIPLOT,x

EOF

for dataset in $datasets; do
cat<<EOF

\begin{tikzpicture}
\begin{axis}[${attributes},
title={Dataset \textsc{${dataset}}},
xlabel={prefix length},
ylabel={total time [s]},
]
%% MULTIPLOT(algo) \$myplot(time_total, "${dataset}")
%% CONFIG file=plot/${dataset}.time_total.tex type=tex
\input{plot/${dataset}.time_total.tex}
\legend{}
\end{axis}
\end{tikzpicture}


\begin{tikzpicture}
\begin{axis}[${attributes},
title={Dataset \textsc{${dataset}}},
xlabel={prefix length},
ylabel={SAT solver time [s]},
]
%% MULTIPLOT(algo) \$myplot(time_total - time_prep, "${dataset}")
%% CONFIG file=plot/${dataset}.time_solve.tex type=tex
\input{plot/${dataset}.time_solve.tex}
\legend{}
\end{axis}
\end{tikzpicture}

\begin{tikzpicture}
\begin{axis}[${attributes},
title={Dataset \textsc{${dataset}}},
xlabel={prefix length},
ylabel={output size},
]
%% MULTIPLOT(algo) \$myplot(factor_size, "${dataset}")
%% CONFIG file=plot/${dataset}.factor_size.tex type=tex
\input{plot/${dataset}.factor_size.tex}
\legend{}
\end{axis}
\end{tikzpicture}

\begin{tikzpicture}
\begin{axis}[${attributes},
title={Dataset \textsc{${dataset}}},
xlabel={prefix length},
ylabel={CNF size},
]
%% MULTIPLOT(algo) \$myplot(sol_ntotalvars, "${dataset}")
%% CONFIG file=plot/${dataset}.sol_ntotalvars.tex type=tex
\input{plot/${dataset}.sol_ntotalvars.tex}
\legend{}
\end{axis}
\end{tikzpicture}

\begin{tikzpicture}
\begin{axis}[${attributes},
title={Dataset \textsc{${dataset}}},
xlabel={prefix length},
ylabel={max.\ clause size},
]
%% MULTIPLOT(algo) \$myplot(sol_nmaxclause, "${dataset}")
%% CONFIG file=plot/${dataset}.sol_nmaxclause.tex type=tex
\input{plot/${dataset}.sol_nmaxclause.tex}
\legend{}
\end{axis}
\end{tikzpicture}

\begin{tikzpicture}
\begin{axis}[${attributes},
title={Dataset \textsc{${dataset}}},
xlabel={prefix length},
ylabel={\# hard clauses},
]
%% MULTIPLOT(algo) \$myplot(sol_nhard, "${dataset}")
%% CONFIG file=plot/${dataset}.sol_nhard.tex type=tex
\input{plot/${dataset}.sol_nhard.tex}
\legend{}
\end{axis}
\end{tikzpicture}
EOF

done
}

function genInclude {
for dataset_it in $(seq 1 $#datasets); do
	dataset=$datasets[$dataset_it]
cat <<EOF
\begin{figure}
	\centering{%
EOF
for i in 0 1 2 3 4 5; do
	((page=(dataset_it-1)*6+4+i))
	echo "\\includegraphics[width=0.4\linewidth,page=${page}]{plot/plot}"
done
cat <<EOF
	}%centering
    \begin{minipage}{0.65\linewidth}
	\caption{Plots for dataset \textsc{$dataset}.}
	\label{plot:${dataset}}
    \end{minipage}
    \hfill
    \begin{minipage}{0.25\linewidth}
        \includegraphics[page=1]{plot/plot}
    \end{minipage}
\end{figure}

EOF
done
}

genPlot > plot_generated.tex

for i in plot/*.tex; do
	sed -i 's@^\\addlegendentry{attr};@\\addlegendentry{$\\gamma$};@' "$i"
	sed -i 's@^\\addlegendentry{bidir};@\\addlegendentry{$b$};@' "$i"
	sed -i 's@^\\addlegendentry{slp};@\\addlegendentry{$g$};@' "$i"
done

