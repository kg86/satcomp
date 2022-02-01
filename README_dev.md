# install minimum string attractor

```bash
# install sdsl
$ git clone https://github.com/simongog/sdsl-lite.git
$ cd sdsl-lite
$ ./install.sh
# install open-wbo
$ git clone https://github.com/sat-group/open-wbo.git
$ cd open-wbo
$ make
$ mv open-wbo ~/.local/bin
```


## run benchmark

```bash
$ pipenv run python src/lz_bench.py --timeout=30 --output=out/hoge.csv --n_jobs 8
```