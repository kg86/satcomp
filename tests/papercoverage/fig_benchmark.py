#!/usr/bin/env python3
"""  """
# pylint: disable=bad-indentation,line-too-long,invalid-name

import subprocess
import sys
import typing
from typing import List
import os
import argparse
from pathlib import Path

MAXTIME = 3600
OUTDIR="."
MAXMEM=4*1024*1024*1024
MAXPREFIX = 1000
MINPREFIX = 50
STEP = 25
LSUTIMEOUT = MAXTIME-600

measures = ["attractor", "bidirectional", "slp"]

algos = { "attractor" : ["attractor_solver"],
		"bidirectional" : ["bms_solver", "bms_fast", "bms_plus"],
		"slp" : ["slp_solver", "slp_fast"]
		}

maxsat_strategy = ['RC2', 'LSU', 'FM']
satsolvers = ['Glucose4', 'Cadical']

def getinstances(filename : str):
	allcombis = ((solver,strategy) for solver in satsolvers for strategy in maxsat_strategy)
	combis = list(filter(lambda pair: (pair[0],pair[1]) != ('Cadical', 'LSU'), allcombis))
	instances = [ (f'{os.path.basename(filename)}_{algo}_{solver}_{strategy}', ['pipenv', 'run', 'python', f'src/{algo}.py', '--file', filename, '--solver', solver, '--strategy', strategy, '--maxtime', MAXTIME, '--maxmem', MAXMEM]) for measure in algos.keys() for algo in algos[measure] for (solver,strategy) in combis]
	return instances
#	return instances + [(f'{os.path.basename(filename)}_{algo}_Glucose4_LSU_approx', ['pipenv', 'run', 'python', f'src/{algo}.py', '--file', filename, '--solver', 'Glucose4', '--strategy', 'LSU', '--maxtime', MAXTIME, '--maxmem', MAXMEM, '--timeout', LSUTIMEOUT]) for measure in algos.keys() for algo in algos[measure]]

def scalingexperiment(filename : str):
	candidates = getinstances(filename)
	isBeaten = {}
	for (candidatename, _) in candidates:
		isBeaten[candidatename] = False
	for prefix in range(MINPREFIX, MAXPREFIX+1, STEP):
		if False not in isBeaten.values():
			break
		for (candidatename, candidatecmd) in candidates:
			if isBeaten[candidatename]:
				continue
			cmd = candidatecmd + ['--prefix', prefix]
			outfilename = f"{OUTDIR}/{candidatename}.{prefix}.json"
			try:
				print(f"Processing {str(cmd)} -> {outfilename}")
				with open(outfilename, "w", encoding='utf-8') as outfile:
					subprocess.run([str(i) for i in cmd], timeout=MAXTIME+300, check=True, stdout=outfile)
			except Exception as e:
				with open(outfilename, "a", encoding='utf-8') as outfile:
					outfile.write(f"{{\"error\": \"{str(e)}\"}}")
				isBeaten[candidatename] = True
				print(f"Error processing {str(cmd)}")
				continue

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Benchmark')
	parser.add_argument('--file', required=True, metavar='filename', type=str, nargs='?', help='input file')
	parser.add_argument('--outdir', required=False, metavar='directory', type=str, nargs='?', help='output directory')
	parser.add_argument('--maxtime', required=False, metavar='time', type=int, nargs='?', help='maximum time')
	parser.add_argument('--maxmem', required=False, metavar='memory', type=int, nargs='?', help='maximum memory')
	parser.add_argument('--maxprefix', required=False, metavar='prefix', type=int, nargs='?', help='maximum prefix')
	parser.add_argument('--minprefix', required=False, metavar='prefix', type=int, nargs='?', help='minimum prefix')
	parser.add_argument('--step', required=False, metavar='step', type=int, nargs='?', help='step')
	parser.add_argument('--fig', required=False, metavar='fig', type=int, nargs='?', help='figure number')

	args = parser.parse_args()
	if not os.path.exists(args.file):
		print(f"File {args.file} does not exist")
		sys.exit(1)
	if args.outdir:
		OUTDIR = args.outdir
	if args.maxtime:
		MAXTIME = args.maxtime
	if args.maxmem:
		MAXMEM = args.maxmem
	if args.maxprefix:
		MAXPREFIX = args.maxprefix
	if args.minprefix:
		MINPREFIX = args.minprefix
	if args.step:
		STEP = args.step
	MAXPREFIX = min(MAXPREFIX, os.path.getsize(args.file))
	if not os.path.exists(OUTDIR):
		os.makedirs(OUTDIR)
	if args.fig == 6:
		maxsat_strategy = ['RC2']
		satsolvers = ['Glucose4']
		scalingexperiment(args.file)
	elif args.fig == 7:
		algos = { "attractor" : [], "bidirectional" : ["bms_solver"], "slp" : ["slp_solver"]}
		maxsat_strategy = ['RC2']
		satsolvers = ['Glucose4']
		scalingexperiment(args.file)
	elif args.fig == 8:
		sys.exit(1)
	else:
		sys.exit(1)
