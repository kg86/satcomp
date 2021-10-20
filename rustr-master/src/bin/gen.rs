// generate various types of words or sequence of words
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "gen", about = "generate various words")]

struct Opt {
    /// Input file, stdin if not present
    #[structopt(short = "k", long, default_value = "5")]
    order: usize,

    #[structopt(long)]
    fibonacci: bool,

    #[structopt(long)]
    fibonacci_plus: bool,

    #[structopt(long)]
    period_doubling: bool,

    #[structopt(long)]
    thue_morse: bool,

    #[structopt(default_value = "", long)]
    of_alphabet: String,

    #[structopt(long)]
    debruijn: bool,
}

fn show_vec(v: &Vec<u8>) {
    println!("{}", std::str::from_utf8(v).unwrap());
}

fn main() {
    let opt = Opt::from_args();
    if opt.fibonacci {
        show_vec(&rustr::words::fibonacci(opt.order));
    }
    if opt.fibonacci_plus {
        show_vec(&rustr::words::fibonacci_plus(opt.order));
    }

    if opt.thue_morse {
        show_vec(&rustr::words::thue_morse(opt.order));
    }
    if opt.period_doubling {
        show_vec(&rustr::words::period_doubling(opt.order));
    }
    if opt.of_alphabet != "" {
        rustr::words::genall_dfs(&opt.of_alphabet.as_bytes(), opt.order);
    }
    if opt.debruijn {}
}
