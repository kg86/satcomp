// output arrays of a string

use std::io;
use std::io::prelude::*;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "arrays", about = "output various arrays of a string")]
struct Opt {
  #[structopt(long)]
  suffix_array: bool,
  #[structopt(long)]
  rank_array: bool,
  #[structopt(long)]
  lcp_array: bool,
}

fn main() {
  let opt = Opt::from_args();
  for line in io::stdin().lock().lines() {
    let s = line.unwrap().as_bytes().to_vec();

    let sa = {
      let mut sa = vec![0; s.len()];
      cdivsufsort::sort_in_place(&s, &mut sa);
      sa
    };
    let rank = rustr::sa::rank_array(&sa);
    let lcp = rustr::sa::lcp_array(&s, &sa, &rank);
    if opt.suffix_array {
      print!("suffix array: ");
      for i in sa {
        print!("{:>4}", i);
      }
      println!("");
    }
    if opt.rank_array {
      print!("rank   array: ");
      for i in rank {
        print!("{:>4}", i);
      }
      println!("");
    }
    if opt.lcp_array {
      print!("lcp    array: ");
      for i in lcp {
        print!("{:>4}", i);
      }
      println!("");
    }
  }
}
