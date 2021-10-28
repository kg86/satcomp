# Computing Smallest Bidirectional Scheme

## Install

```bash
$ pip install pipenv
$ pipenv sync
```

## Usage

```bash
$ pipenv run python bidirectional_solver.py                                         master
usage: bidirectional_solver.py [-h] [--file FILE]

Compute Minimum Bidirectional Scheme

optional arguments:
  -h, --help   show this help message and exit
  --file FILE  input file
```

## Running Example

```bash
$ pipenv run python bidirectional_solver.py  --file data/cantrbry_pref/cp.html-50
[520 -   <module>() ] b'<head>\n<title>Compression Pointers</title>\n<META H'
[233 - bidirectional_WCNF() ] bidirectional_solver start
[234 - bidirectional_WCNF() ] # of text = 50, # of lz77 = 44
[276 - bidirectional_WCNF() ] 35/50 occurs only once
[391 - bidirectional_WCNF() ] #literals = 1140, # hard clauses=3517, # of soft clauses=50
[441 - bidirectional() ] # of [BiDirLiteral.true] literals  = 1
[441 - bidirectional() ] # of [BiDirLiteral.false] literals  = 1
[441 - bidirectional() ] # of [BiDirLiteral.link_to] literals  = 368
[441 - bidirectional() ] # of [BiDirLiteral.fbeg] literals  = 50
[441 - bidirectional() ] # of [BiDirLiteral.root] literals  = 50
[441 - bidirectional() ] # of [BiDirLiteral.ref] literals  = 92
[441 - bidirectional() ] # of [BiDirLiteral.depth_ref] literals  = 200
[441 - bidirectional() ] # of [BiDirLiteral.auxlit] literals  = 148
[535 -   <module>() ] runtime: 0.024665117263793945
[536 -   <module>() ] len=43: factors=[(-1, 60), (-1, 104), (-1, 101), (-1, 97), (-1, 100), (41, 3), (36, 6), (-1, 67), (-1, 111), (-1, 109), (-1, 112), (-1, 114), (-1, 101), (-1, 115), (-1, 115), (-1, 105), (-1, 111), (-1, 110), (-1, 32), (-1, 80), (-1, 111), (-1, 105), (-1, 110), (-1, 116), (-1, 101), (-1, 114), (-1, 115), (-1, 60), (-1, 47), (-1, 116), (-1, 105), (-1, 116), (-1, 108), (-1, 101), (-1, 62), (-1, 10), (-1, 60), (-1, 77), (-1, 69), (-1, 84), (-1, 65), (-1, 32), (-1, 72)]
len of text = 50
decode=b'<head>\n<title>Compression Pointers</title>\n<META H'
equals original? True
{"date": "2021-10-25 17:02:30.310772", "status": "", "algo": "solver", "file_name": "cp.html-50", "file_len": 50, "time_prep": 0.01598215103149414, "time_total": 0.024064064025878906, "bd_size": 43, "bd_factors": [[-1, 60], [-1, 104], [-1, 101], [-1, 97], [-1, 100], [41, 3], [36, 6], [-1, 67], [-1,
111], [-1, 109], [-1, 112], [-1, 114], [-1, 101], [-1, 115], [-1, 115], [-1, 105], [-1, 111], [-1, 110], [-1, 32], [-1, 80], [-1, 111], [-1, 105], [-1, 110], [-1, 116], [-1, 101], [-1, 114], [-1, 115], [-1, 60], [-1, 47], [-1, 116], [-1, 105], [-1, 116], [-1, 108], [-1, 101], [-1, 62], [-1, 10], [-1, 60], [-1, 77], [-1, 69], [-1, 84], [-1, 65], [-1, 32], [-1, 72]], "sol_nvars": 1140, "sol_nhard": 3517, "sol_nsoft": 50}
```

## For Development

run benchmark.

```bash
$ pipenv run python bidirectional_bench.py
```

convert json to csv.

```bash
$ json2csv -k date,status,algo,file_name,file_len,time_prep,time_total,bd_size,bd_factors,sol_nvars,sol_nhard,sol_nsoft -i exp_timeout=60.txt -p > hoge.csv
```