use rustr::bwt::*;
use std::io;
use std::io::prelude::*;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
  #[structopt(long)]
  echo_input: bool,

  #[structopt(long)]
  reverse_input: bool,

  #[structopt(long)]
  bwt: bool,

  #[structopt(long)]
  bbwt: bool,

  #[structopt(long)]
  lz77: bool,

  #[structopt(long)]
  lyndon_factorization: bool,
}

fn main() {
  let opt = Opt::from_args();
  for line in io::stdin().lock().lines() {
    let mut l = line.unwrap().as_bytes().to_vec();
    if opt.reverse_input {
      l.reverse();
    }
    println!("{{");
    if opt.echo_input {
      println!("\tinput: \"{}\"", String::from_utf8(l.clone()).unwrap());
    }
    if opt.bwt {
      let t = bw_transform(&l);
      println!("\tbwt: \"{}\"", std::str::from_utf8(&t).unwrap());
      println!("\trlbwtsize: {}", rustr::rle::runlength(&t));
    }
    if opt.bbwt {
      let t = rustr::rle::encode(&bbw_transform(&l));
      println!("\tbbwt: \"{:?}\"", t); //String::from_utf8(t).unwrap());
      println!("\trlbbwtsize: {}", t.len());
    }
    if opt.lz77 {
      let t = rustr::lz::lz77(&l);
      println!("\tlz77: \"{:?}\"", t);
      println!("\tlz77size: {}", t.len());
    }
    println!("}}");
  }
}
