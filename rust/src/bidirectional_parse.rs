// routines for bidirectional macro schemes
use cdivsufsort::sort_in_place;

#[derive(Copy, Clone, Debug)]
pub enum BDPhrase {
    Source { len: u32, pos: u32 },
    Ground(u8),
}

pub fn decode(parse: &[BDPhrase]) -> Option<Vec<u8>> {
    let totlen: usize = parse.iter().fold(0, |sum, phrase| match phrase {
        BDPhrase::Ground(_) => sum + 1_usize,
        BDPhrase::Source { len: l, .. } => sum + *l as usize,
    });
    let mut res = vec![0; totlen];
    let mut source = vec![0u32; totlen];
    let mut cur_pos = 0usize;
    for phrase in parse.iter() {
        match phrase {
            BDPhrase::Ground(c) => {
                source[cur_pos] = cur_pos as u32;
                res[cur_pos] = *c;
                cur_pos += 1;
            }
            BDPhrase::Source { len: l, pos: p } => {
                if *p == cur_pos as u32 {
                    return None;
                }
                for i in 0..*l {
                    source[cur_pos] = p + i;
                    cur_pos += 1
                }
            }
        }
    }

    // Kahn's topological sort algorithm
    let mut indegree = vec![0_u32; totlen];
    for (i, s) in source.iter().enumerate().take(totlen) {
        if *s as usize != i {
            indegree[*s as usize] += 1
        }
    }
    let mut stack = Vec::new();
    for (i, d) in indegree.iter().enumerate().take(totlen) {
        if *d == 0 {
            stack.push(i)
        }
    }

    let mut cnt = 0;
    let mut top_order = Vec::new();
    while !stack.is_empty() {
        let u = stack.pop().unwrap();
        top_order.push(u);
        if source[u] as usize != u {
            indegree[source[u] as usize] -= 1;
            if indegree[source[u] as usize] == 0 {
                stack.push(source[u] as usize);
            }
        }
        cnt += 1;
    }
    if cnt != totlen {
        return None;
    }
    for p in top_order.iter().rev() {
        if source[*p] as usize != *p {
            res[*p] = res[source[*p] as usize];
        }
    }
    //println!("Reconstructed: {}", std::str::from_utf8(&res).unwrap());
    Some(res)
}

// find combination of valid sources of itv, given its vector of occs
fn find_valid_sources_dfs(
    s: &[u8],
    itv: &Vec<&[u8]>,
    occs: &Vec<Vec<usize>>,
    phrases: &mut Vec<BDPhrase>,
) -> Option<Vec<BDPhrase>> {
    if phrases.len() == itv.len() {
        //println!("{:?}", phrases);
        if decode(phrases).is_some() {
            return Some(phrases.clone());
        }
    } else {
        let next_itv = itv[phrases.len()];
        if next_itv.len() == 1 {
            phrases.push(BDPhrase::Ground(next_itv[0]));
            match find_valid_sources_dfs(s, itv, occs, phrases) {
                None => (),
                x => return x,
            }
            phrases.pop();
        } else {
            for occ in occs[phrases.len()].iter() {
                let next_phrase = match next_itv.len() {
                    1 => BDPhrase::Ground(s[*occ]),
                    _ => BDPhrase::Source {
                        len: next_itv.len() as u32,
                        pos: *occ as u32,
                    },
                };
                phrases.push(next_phrase);
                match find_valid_sources_dfs(s, itv, occs, phrases) {
                    None => (),
                    x => return x,
                }
                phrases.pop();
            }
        }
    }
    None
}

fn find_other_occurrences(esa: &crate::sa::ESA, cur_len: usize, l: usize) -> Vec<usize> {
    let mut ocs = Vec::new();
    let mut i = esa.rank[cur_len];
    while i > 0 && esa.lcp[i as usize] as usize >= l {
        ocs.push(esa.sa[i as usize - 1] as usize);
        i -= 1;
    }
    let mut i = esa.rank[cur_len];
    while i as usize + 1 < esa.sa.len() && esa.lcp[i as usize + 1] as usize >= l {
        ocs.push(esa.sa[i as usize + 1] as usize);
        i += 1;
    }
    ocs.sort_unstable();
    ocs
}

// aux function for finding a parsing of size k > 0
fn enum_factorizations_dfs<'a>(
    s: &'a [u8],
    esa: &crate::sa::ESA,
    cur_len: usize,
    k: usize,
    first_phrase_len: usize,
    occs: &mut Vec<Vec<usize>>,
    itv: &mut Vec<&'a [u8]>,
) -> Option<Vec<BDPhrase>> {
    if k > s.len() {
        return None;
    }
    if itv.len() == 3 {
        for i in itv.iter() {
            print!("{};", i.len());
        }
        println!();
    }
    if k == 0 {
        let mut p = Vec::new();
        return find_valid_sources_dfs(s, itv, occs, &mut p);
    } else if k == 1 {
        let l = s.len() - cur_len;
        let next_itv = &s[cur_len..s.len()];
        let ocs = if l > 1 {
            find_other_occurrences(esa, cur_len, l)
        } else {
            Vec::new()
        };
        if l == 1 || !ocs.is_empty() {
            itv.push(next_itv);
            occs.push(ocs);
            let res =
                enum_factorizations_dfs(s, esa, cur_len + l, k - 1, first_phrase_len, occs, itv);
            match res {
                None => (),
                x => return x,
            }
            occs.pop();
            itv.pop();
        }
    } else {
        let maxl = std::cmp::max(1, {
            let i = esa.rank[cur_len] as usize;
            if i + 1 < esa.rank.len() {
                std::cmp::max(esa.lcp[i], esa.lcp[i + 1]) as usize
            } else {
                esa.lcp[i] as usize
            }
        });
        let rng = if first_phrase_len > 0 && cur_len == 0 {
            first_phrase_len..first_phrase_len + 1
        } else {
            1..std::cmp::min(s.len() - cur_len - k + 2, maxl + 1)
        };
        for l in rng {
            //for l in 1..s.len() - cur_len - k + 2 {
            let next_itv = &s[cur_len..cur_len + l];
            let ocs = if l > 1 {
                find_other_occurrences(esa, cur_len, l)
            } else {
                Vec::new()
            };
            if l == 1 || !ocs.is_empty() {
                itv.push(next_itv);
                occs.push(ocs);
                let res = enum_factorizations_dfs(
                    s,
                    esa,
                    cur_len + l,
                    k - 1,
                    first_phrase_len,
                    occs,
                    itv,
                );
                match res {
                    None => (),
                    x => return x,
                }
                occs.pop();
                itv.pop();
            }
        }
    }
    None
}

// find a parsing of size k for string s
fn find_of_size_aux(
    s: &[u8],
    sa: &crate::sa::ESA,
    first_phrase_len: usize,
    k: usize,
) -> Option<Vec<BDPhrase>> {
    let mut itv = Vec::new();
    let mut occs = Vec::new();
    enum_factorizations_dfs(s, sa, 0, k, first_phrase_len, &mut occs, &mut itv)
}

pub fn find_of_size(s: &[u8], first_phrase_len: usize, k: usize) -> Option<Vec<BDPhrase>> {
    let sa = {
        let mut sa = vec![0; s.len()];
        sort_in_place(s, &mut sa);
        sa
    };
    let rank = crate::sa::rank_array(&sa);
    let lcp = crate::sa::lcp_array(s, &sa, &rank);
    find_of_size_aux(
        s,
        &crate::sa::ESA {
            sa: &sa,
            rank: &rank,
            lcp: &lcp,
        },
        first_phrase_len,
        k,
    )
}

pub fn find_in_range(
    s: &[u8],
    minsz: usize,
    maxsz: usize,
    first_phrase_len: usize,
) -> Option<Vec<BDPhrase>> {
    if !s.is_empty() {
        for k in minsz..std::cmp::min(maxsz, s.len()) + 1 {
            // k is the number of factors
            println!("Checking for bidirectional parse of size: {}", k);
            match find_of_size(s, first_phrase_len, k) {
                None => (),
                x => return x,
            }
        }
    }
    None
}

pub fn find_at_least(s: &[u8], minsz: usize, first_phrase_len: usize) -> Vec<BDPhrase> {
    let res = Vec::new();
    if !s.is_empty() {
        match find_in_range(s, minsz, s.len(), first_phrase_len) {
            None => (),
            Some(x) => return x,
        }
    }
    res
}

/// find smallest bidirectional parse of string s
pub fn find_optimal(s: &[u8], first_phrase_len: usize) -> Vec<BDPhrase> {
    find_at_least(s, 1, first_phrase_len)
}

#[test]
fn test_find_optimal() {
    let s = "ab".as_bytes();
    let r = find_at_least(&s, 2, 0);
    assert_eq!(r.len(), 2);
    let i = 3;
    //let s = crate::words::period_doubling(i); // crate::words::thuemorse(i);
    let s = crate::words::thue_morse(i);
    let r = find_optimal(&s, 0);
    //println!("{:?}", r);
    assert_eq!(r.len(), i + 2);
}

#[test]
fn test_decode() {
    let b = vec![
        BDPhrase::Source { len: 4, pos: 6 },
        BDPhrase::Ground(b'b'),
        BDPhrase::Ground(b'a'),
        BDPhrase::Source { len: 2, pos: 3 },
        BDPhrase::Source { len: 4, pos: 4 },
        BDPhrase::Source { len: 4, pos: 6 },
    ];
    let r = decode(&b);
    assert_eq!(r.unwrap(), crate::words::thue_morse(4));
    assert_eq!(
        decode(&vec![
            BDPhrase::Source { len: 30, pos: 32 },
            BDPhrase::Source { len: 5, pos: 52 },
            BDPhrase::Source { len: 2, pos: 14 },
            BDPhrase::Ground(98),
            BDPhrase::Source { len: 10, pos: 54 },
            BDPhrase::Source { len: 15, pos: 24 },
            BDPhrase::Ground(97)
        ])
        .unwrap(),
        crate::words::period_doubling(6)
    );
}
