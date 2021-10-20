// use libdivsufsort_rs::*;
use cdivsufsort::sort_in_place;
// use std::collections::HashMap;

// Lempel-Ziv parsings
#[derive(Debug, PartialEq)]

/// a structure for representing the "source" of an LZ77 phrase (position, or ground char)
pub enum CharOrSource {
    Source(usize),
    Char(u8),
}

/*
struct TrieNode {
  c: HashMap<u8, TrieNode>,
}

// LZ 78 parsing : f_0, f_1, ..., (f_0 is empty string)
// where f_i = f_k c is the longest prefix of f_i ... for some k < i.
#[allow(dead_code)]
pub fn lz78(s: &Vec<u8>) -> Vec<TrieNode> {
  let res = Vec::new();
  let mut i = 0;
  let mut l = 0;
  let root = TrieNode { c: HashMap::new() };
  let mut node = &root;
  while i < s.len() {
    l = 0;
    loop {
      if i >= s.len() {
        break;
      }
      i += 1;
      l += 1;
      match node.c.get(&s[i - 1]) {
        Some(child) => {
          node = child;
        }
        None => break,
      }
    }
    if l > 1 {
      res.push(CharOrSource::Source)
    }
  }
  res
}
*/

// LZ 77 parsing : f_0, f_1, ...
// where f_i is the longest prefix of f_i ... that occurs twice in f_0 ... f_i.
pub fn naivelz77(s: &[u8]) -> Vec<(usize, CharOrSource)> {
    let mut res = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let mut source = CharOrSource::Char(s[i]);
        let mut maxl = 0;
        for j in 0..i {
            let mut l = 0;
            while i + l < s.len() {
                if s[j + l] != s[i + l] {
                    break;
                }
                l = l + 1;
            }
            if l > maxl && l > 1 {
                maxl = l;
                source = CharOrSource::Source(j);
            }
        }
        res.push((maxl, source));
        i = i + std::cmp::max(1, maxl);
    }
    res
}

// index of previous smaller value in array
fn psv(a: &[i32]) -> Vec<i32> {
    let mut res = vec![-1 as i32; a.len()];
    for i in 0..a.len() {
        let mut p = i as i32 - 1;
        while p >= 0 && a[p as usize] >= a[i] {
            p = res[p as usize];
        }
        res[i] = p;
    }
    res
}

// index of next smaller value in array
fn nsv(a: &[i32]) -> Vec<i32> {
    let mut res = vec![a.len() as i32; a.len()];
    for i in (0..a.len()).rev() {
        let mut p = i as i32 + 1;
        while p < a.len() as i32 && a[p as usize] >= a[i] {
            p = res[p as usize];
        }
        res[i] = p;
    }
    res
}

// assume p1 > p2 and return 0 if p2 == -1
fn naive_lcp(s: &[u8], p1: i32, p2: i32) -> i32 {
    if p2 < 0 || p2 >= s.len() as i32 {
        0
    } else {
        let mut l = 0;
        while ((p1 + l) as usize) < s.len()
            && ((p2 + l) as usize) < s.len()
            && s[(p1 + l) as usize] == s[(p2 + l) as usize]
        {
            l = l + 1;
        }
        l
    }
}

pub fn lz77(s: &[u8]) -> Vec<(usize, CharOrSource)> {
    let mut res = Vec::new();
    let sa = {
        let mut sa = vec![0; s.len()];
        sort_in_place(s, &mut sa);
        sa
    };
    let rank = crate::sa::rank_array(&sa);
    let sa_psv = psv(&sa);
    let sa_nsv = nsv(&sa);
    res.push((0, CharOrSource::Char(s[0])));
    let mut p: i32 = 1;
    while (p as usize) < s.len() {
        let mut source = sa_psv[rank[p as usize] as usize];
        let mut l = if source < 0 {
            0
        } else {
            source = sa[source as usize];
            naive_lcp(&s, p, source)
        };
        let sn = sa_nsv[rank[p as usize] as usize];
        let nlen = if sn as usize >= s.len() {
            0
        } else {
            naive_lcp(&s, p, sa[sn as usize])
        };
        if nlen > l {
            source = sa[sn as usize];
            l = nlen;
        }
        if l < 2 {
            res.push((0, CharOrSource::Char(s[p as usize])));
            p = p + 1;
        } else {
            res.push((l as usize, CharOrSource::Source(source as usize)));
            p = p + l;
        }
    }
    res
}

pub fn lz77_count(s: &[u8]) -> usize {
    let mut res = 0;
    let sa = {
        let mut sa = vec![0; s.len()];
        sort_in_place(s, &mut sa);
        sa
    };
    let rank = crate::sa::rank_array(&sa);
    let sa_psv = psv(&sa);
    let sa_nsv = nsv(&sa);
    if s.len() > 0 {
        res += 1;
        let mut p: i32 = 1;
        while (p as usize) < s.len() {
            let mut source = sa_psv[rank[p as usize] as usize];
            let mut l = if source < 0 {
                0
            } else {
                source = sa[source as usize];
                naive_lcp(&s, p, source)
            };
            let sn = sa_nsv[rank[p as usize] as usize];
            let nlen = if sn as usize >= s.len() {
                0
            } else {
                naive_lcp(&s, p, sa[sn as usize])
            };
            if nlen > l {
                l = nlen;
            }
            if l < 2 {
                res += 1;
                p = p + 1;
            } else {
                res += 1;
                p = p + l;
            }
        }
    }
    res
}

#[test]
fn test_lz77() {
    let s = "aaaaaaaaaaaaaaaaaaaaaaaaaa".as_bytes().to_vec();
    let x = naivelz77(&s);
    //println!("{:?}", x);
    let y = lz77(&s);
    assert_eq!(x, y);
    assert_eq!(lz77_count(&s), 2);
}
