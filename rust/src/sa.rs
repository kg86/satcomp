// routines related to suffix arrays

pub struct ESA<'a> {
    pub sa: &'a [i32],
    pub rank: &'a [i32],
    pub lcp: &'a [i32],
}

/// compute rank array, given suffix array
pub fn rank_array(sa: &[i32]) -> Vec<i32> {
    let mut res = vec![0_i32; sa.len()];
    for i in 0..sa.len() {
        res[sa[i] as usize] = i as i32;
    }
    res
}

/// compute lcp array given suffix and rank arrays via Kasai's algorithm
pub fn lcp_array(s: &[u8], sa: &[i32], rank: &[i32]) -> Vec<i32> {
    let mut lcp = vec![0; s.len()];
    let mut k = 0;
    for i in 0..s.len() {
        let x = rank[i];
        if x > 0 {
            let y = sa[x as usize - 1] as usize;
            while i + k < s.len() && y + k < s.len() && s[i + k] == s[y + k] {
                k += 1;
            }
            lcp[x as usize] = k as i32;
        }
        k = k.saturating_sub(1);
    }
    //println!("{:?}", lcp);
    lcp
}

#[test]
fn test_lcp_array() {
    let s = crate::words::thue_morse(3);
    let sa = {
        let mut sa = vec![0; s.len()];
        cdivsufsort::sort_in_place(&s, &mut sa);
        sa
    };
    let rank = crate::sa::rank_array(&sa);
    let lcp = crate::sa::lcp_array(&s, &sa, &rank);
    assert_eq!(lcp, vec![0, 1, 2, 2, 0, 1, 2, 1]);
}
