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
MAXPREFIX = 100000

measures = ["attractor", "bidirectional", "slp"]

algos = { "attractor" : ["attractor_solver"],
		"bidirectional" : ["bms_solver", "bms_fast", "bms_plus"],
		"slp" : ["slp_solver", "slp_fast"]
		}

maxsat_strategy = ['RC2', 'LSU', 'FM']

def getinstances(filename : str):
	return [ (f'{os.path.basename(filename)}_{algo}_{strategy}', ['pipenv', 'run', 'python', f'src/{algo}.py', '--file', filename, '--strategy', strategy, '--maxtime', MAXTIME, '--maxmem', MAXMEM]) for measure in algos.keys() for algo in algos[measure] for strategy in maxsat_strategy]

def scalingexperiment(filename : str):
	candidates = getinstances(filename)
	isBeaten = {}
	for (candidatename, _) in candidates:
		isBeaten[candidatename] = False
	for prefix in range(10, MAXPREFIX, 10):
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
	MAXPREFIX = min(MAXPREFIX, os.path.getsize(args.file))
	if not os.path.exists(OUTDIR):
		os.makedirs(OUTDIR)
	scalingexperiment(args.file)
