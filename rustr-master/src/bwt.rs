//
// Burrows-Wheeler transform based on cyclic rotations
//

use std::cmp::Ordering;

// compute conjugate of s starting at position i
#[allow(dead_code)]
fn conjugate(s: &[u8], i: usize) -> Vec<u8> {
    let mut res = Vec::with_capacity(s.len());
    for j in 0..s.len() {
        res.push(s[(i + j) % s.len()]);
    }
    res
}

// compute bwt array (lexicographically sorting conjugates, not suffixes)
pub fn bwt_array(s: &[u8]) -> Vec<i32> {
    let mut sa = (0..s.len() as i32).collect::<Vec<i32>>();
    sa.sort_by(|a, b| {
        let mut res = Ordering::Equal;
        for i in 0..s.len() {
            if s[((*a as usize + i) % s.len())] < s[(*b as usize + i) % s.len()] {
                res = Ordering::Less;
                break;
            } else if s[(*a as usize + i) % s.len()] > s[(*b as usize + i) % s.len()] {
                res = Ordering::Greater;
                break;
            }
        }
        if res != Ordering::Equal {
            res
        } else if a < b {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });
    sa
}

fn omega_order(s1: &[u8], i1: usize, s2: &[u8], i2: usize) -> Ordering {
    let mut res = Ordering::Equal;
    for i in 0..std::cmp::max(s1.len(), s2.len()) * 2 {
        let c1 = s1[(i1 + i) % s1.len()];
        let c2 = s2[(i2 + i) % s2.len()];
        if c1 < c2 {
            res = Ordering::Less;
            break;
        } else if c1 > c2 {
            res = Ordering::Greater;
            break;
        }
    }
    res
}

// the bw transform for a multiset of primitive strings
pub fn ebw_transform(ss: &[&[u8]]) -> Vec<u8> {
    let mut sa = Vec::new();
    for s in 0..ss.len() {
        for i in 0..ss[s].len() {
            sa.push((ss[s], i));
        }
    }
    sa.sort_by(|(a_s, a_i), (b_s, b_i)| omega_order(a_s, *a_i, b_s, *b_i));
    let res = sa
        .into_iter()
        .map(|(s, i)| s[(i + s.len() - 1) % s.len()])
        .collect();
    res
}

pub fn show_sorted_conjugates(ss: &[&[u8]]) {
    let mut sa = Vec::new();
    for s in 0..ss.len() {
        for i in 0..ss[s].len() {
            sa.push((ss[s], i));
        }
    }
    sa.sort_by(|(a_s, a_i), (b_s, b_i)| omega_order(a_s, *a_i, b_s, *b_i));
    for (s, i) in sa.iter() {
        println!(
            "{}",
            String::from_utf8(conjugate(&s.to_vec(), *i).clone()).unwrap()
        )
    }
}

// the bw transform based on bwt_array
// (or suffix array with lex smallest $ - which must be guaranteed by the caller)
pub fn bw_transform(s: &[u8]) -> Vec<u8> {
    let mut res = Vec::with_capacity(s.len());
    let sa = bwt_array(&s);
    for i in 0..s.len() {
        res.push(s[(sa[i] as usize + s.len() - 1) % s.len()])
    }
    res
}

// the bijective BW transform
pub fn bbw_transform(s: &[u8]) -> Vec<u8> {
    let ss = crate::lyndon::decomposed_factorization(&s);
    show_sorted_conjugates(&ss);
    ebw_transform(&ss)
}

// The inverse bwt transform which generates a set of cyclic strings
pub fn inverse_bwt(bwt: &[u8]) -> Vec<Vec<u8>> {
    let mut res = Vec::new();
    let mut prevcounts = vec![0; 255];
    for i in 0..bwt.len() {
        prevcounts[bwt[i] as usize] += 1;
    }
    let mut counts = prevcounts.clone();
    prevcounts[0] = 0;
    for i in 1..255 {
        prevcounts[i] = prevcounts[i - 1] + counts[i - 1];
    }
    let mut lfmap = vec![0; bwt.len()];
    for i in (0..bwt.len()).rev() {
        counts[bwt[i] as usize] -= 1;
        lfmap[i] = prevcounts[bwt[i] as usize] + counts[bwt[i] as usize];
    }
    let mut used = vec![false; bwt.len()];
    let mut i = 0;
    while i < bwt.len() {
        if used[i] {
            i += 1;
        } else {
            let mut k = i;
            let mut w = Vec::new();
            loop {
                w.push(bwt[k]);
                used[k] = true;
                k = lfmap[k];
                if k == i {
                    break;
                }
            }
            w.reverse();
            res.push(w);
        }
    }
    res
}

fn is_reducible_lcp(s: &[u8], sa: &[i32], i: usize) -> bool {
    i > 0
        && s[(sa[i - 1] as usize + s.len() - 1) % s.len()]
            == s[(sa[i] as usize + s.len() - 1) % s.len()]
}

pub fn ilcp(sv: &[u8]) -> Vec<i32> {
    let sa = bwt_array(&sv);
    let rank = crate::sa::rank_array(&sa);
    let lcp = crate::sa::lcp_array(&sv, &sa, &rank);
    let mut irreducible_lcps = Vec::new();
    for i in 0..sv.len() {
        if !is_reducible_lcp(&sv, &sa, i) {
            if lcp[i] > 0 {
                irreducible_lcps.push(lcp[i]);
            }
        }
    }
    println!("irreducible lcps: {:?}", irreducible_lcps);
    irreducible_lcps
}

#[test]
fn test_bwt() {
    let s = "abaababaabaababaababa".as_bytes().to_vec();
    let bwt = bw_transform(&s);
    println!("{}", String::from_utf8(bwt.clone()).unwrap());
    let ibwt = inverse_bwt(&bwt);
    for x in ibwt.iter() {
        println!(":{}", String::from_utf8(x.to_vec().clone()).unwrap());
    }
}

#[test]
fn test_bbwt() {
    let s = "abaababaabaababaababaabaababaabaab".as_bytes().to_vec();
    assert_eq!(
        bbw_transform(&s),
        "bbbbbbbbaabbababaaaabaaaaaaaaaaaaa".as_bytes().to_vec()
    );
}
