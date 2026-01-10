use satcomp::bms_parse::BDPhrase;

use std::{fs, io};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "optimal_bms",
    about = "find optimal bidirectional macro scheme (BMS)"
)]
struct Opt {
    /// Input file, stdin if not present
    #[structopt(short = "k", long, default_value = "1")]
    minsize: usize,

    #[structopt(long = "maxsize")]
    maxsize: Option<usize>,

    #[structopt(long, default_value = "0")]
    first_phrase_len: usize,

    #[structopt(long = "input_file")]
    input_file: String,
}

fn serialize_bd(r: Vec<BDPhrase>) -> Vec<(i32, i32)> {
    let mut res: Vec<(i32, i32)> = Vec::new();
    // let x = (0, 2);
    for item in r.into_iter() {
        res.push(match item {
            BDPhrase::Source { len, pos } => (pos as i32, len as i32),
            BDPhrase::Ground(c) => (-1, c as i32),
        });
    }
    res
}

fn main() -> io::Result<()> {
    let opt = Opt::from_args();

    // for line in io::stdin().lock().lines() {
    //     let line = line.unwrap().as_bytes().to_vec();
    //     let r = rustr::bms_parse::find_in_range(
    //         &line,
    //         opt.minsize,
    //         match opt.maxsize {
    //             None => usize::MAX,
    //             Some(x) => x,
    //         },
    //         opt.first_phrase_len,
    //     );
    //     println!("{:?}", r);
    //     if let Some(bd) = r {
    //         println!("{:?}", serialize_bd(bd))
    //     }
    // }
    // let args: Vec<String> = env::args().collect();
    // println!("{:?}", args);
    let text = fs::read(&opt.input_file)?;
    let r = satcomp::bms_parse::find_in_range(
        &text,
        opt.minsize,
        opt.maxsize.map_or(usize::MAX, |x| x),
        opt.first_phrase_len,
    );
    if let Some(bd) = r {
        println!("{:?}", serialize_bd(bd))
    }
    Ok(())
}
