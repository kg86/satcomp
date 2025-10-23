#!/usr/bin/env python3
""" concatenates contents of JSON files to a JSON list """

import sys
import json
from pathlib import Path

l=[]
for file in sys.argv[1:]:
    s = Path(file).read_text(encoding = 'utf-8')
    if s:
        try:
            l.append(json.loads(s))
        except json.JSONDecodeError as e:
            print(e)
            print("file: " + file, file=sys.stderr)
            raise e


print(json.dumps(l))
