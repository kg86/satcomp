use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(name = "slp_naive", about = "A program of slp_naive.")]
struct Args {
    /// An input file.
    #[clap(short = 'f', long)]
    file: Option<String>,

    /// An input text.
    #[clap(short = 't', long)]
    text: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let text = if let Some(file) = args.file {
        fs::read_to_string(&file)
            .with_context(|| format!("Failed to read the input file {}", &file))?
    } else if let Some(text) = args.text {
        text
    } else {
        Err(anyhow!("Failed to parse {:?}", args))?
    };

    let text = text.as_bytes();
    let min_size = Cell::new(text.len() * 2);

    let start = Instant::now();
    enum_ordered(text, &|root: Node| {
        let mut node_map = HashMap::new();
        minimize_tree(Rc::new(root), &mut node_map);
        min_size.set(min_size.get().min(node_map.len()));
    });
    let duration = start.elapsed();

    println!("minimum SLP size = {}", min_size.get());
    println!("elapsed time = {} sec", duration.as_secs_f64());

    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Node {
    Inner(InnerNode),
    Leaf(LeafNode),
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Inner(x) => x.fmt(f),
            Node::Leaf(x) => x.fmt(f),
        }
    }
}

#[derive(Clone, Debug)]
struct InnerNode {
    left: Option<Rc<Node>>,
    right: Option<Rc<Node>>,
}

impl fmt::Display for InnerNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}{})",
            self.left.as_ref().unwrap(),
            self.right.as_ref().unwrap()
        )
    }
}

impl PartialEq for InnerNode {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(self.left.as_ref().unwrap(), other.left.as_ref().unwrap())
            && Rc::ptr_eq(self.right.as_ref().unwrap(), other.right.as_ref().unwrap())
    }
}

impl Eq for InnerNode {}

impl Hash for InnerNode {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        hasher.write_usize(Rc::as_ptr(self.left.as_ref().unwrap()) as usize);
        hasher.write_usize(Rc::as_ptr(self.right.as_ref().unwrap()) as usize);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LeafNode {
    label: u8,
}

impl fmt::Display for LeafNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label as char)
    }
}

// cf. http://www.nct9.ne.jp/m_hiroi/linux/rust03.html
fn enum_ordered(labels: &[u8], f: &dyn Fn(Node)) {
    if labels.len() == 1 {
        f(Node::Leaf(LeafNode { label: labels[0] }));
    } else {
        for i in 0..labels.len() {
            enum_ordered(&labels[..i], &|left: Node| {
                enum_ordered(&labels[i..], &|right: Node| {
                    f(Node::Inner(InnerNode {
                        left: Some(Rc::new(left.clone())),
                        right: Some(Rc::new(right)),
                    }))
                });
            });
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MapKey {
    Inner((Rc<Node>, Rc<Node>)),
    Leaf(Rc<Node>),
}

fn minimize_tree(root: Rc<Node>, node_map: &mut HashMap<MapKey, Rc<Node>>) -> Rc<Node> {
    match root.as_ref() {
        Node::Inner(inner) => {
            let left = minimize_tree(Rc::clone(inner.left.as_ref().unwrap()), node_map);
            let right = minimize_tree(Rc::clone(inner.right.as_ref().unwrap()), node_map);
            let key = MapKey::Inner((left, right));
            #[allow(clippy::option_if_let_else)]
            if let Some(k) = node_map.get(&key) {
                Rc::clone(k)
            } else {
                node_map.insert(key, Rc::clone(&root));
                root
            }
        }
        Node::Leaf(_) => {
            let key = MapKey::Leaf(Rc::clone(&root));
            node_map.entry(key).or_insert_with(|| Rc::clone(&root));
            root
        }
    }
}
