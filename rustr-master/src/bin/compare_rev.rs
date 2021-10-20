// compare various measurements between forward and reverse strings
use std::io::prelude::*;
// use std::io::{self, BufReader};
use std::io;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
  name = "compare_rev",
  about = "compare various measurements between forward and reverse strings."
)]
struct Opt {
  /// Input file, stdin if not present
  #[structopt(parse(from_os_str))]
  input: Option<PathBuf>,

  /// Output file, stdout if not present
  #[structopt(parse(from_os_str))]
  output: Option<PathBuf>,
}

fn main() -> io::Result<()> {
  let _opt = Opt::from_args();
  for line in io::stdin().lock().lines() {
    let line = line.unwrap().as_bytes().to_vec();
    let mut rline = line.clone();
    rline.reverse();
    print!("len: {:8}", line.len());
    let v = rustr::rle::runlength(&rustr::bwt::bw_transform(&line));
    let v1 = rustr::rle::runlength(&rustr::bwt::bw_transform(&rline));
    let rat: f64 = v as f64 / v1 as f64;
    let rat = if rat < 1.0 { 1.0 / rat } else { rat };
    print!("\t[bwt: {}/{} = {:.2}]", v, v1, rat);
    let v = rustr::rle::runlength(&rustr::bwt::bbw_transform(&line));
    let v1 = rustr::rle::runlength(&rustr::bwt::bbw_transform(&rline));
    let rat: f64 = v as f64 / v1 as f64;
    let rat = if rat < 1.0 { 1.0 / rat } else { rat };
    println!("\t[bbwt: {}/{} = {:.2}]", v, v1, rat);
  }
  Ok(())
}
