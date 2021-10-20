//use std::str;
// Duval's algorithm: output's a vector of slices and exponent
/*
pub fn factorization(s: &Vec<u8>) -> Vec<(&[u8], usize)> {
  let mut res = Vec::new();
  let (mut i, mut j, mut l) = (0, 1, 0);
  while j + l < s.len() {
    if s[i + l] == s[j + l] {
      l += 1;
    } else if s[i + l] < s[j + l] {
      res.push();
      j = j + l + 1;
      l = 0;
    }
  }
  res
}
*/

pub fn decomposed_factorization(s: &[u8]) -> Vec<&[u8]> {
    let mut res = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let (mut j, mut k) = (i + 1, i);
        while j < s.len() && s[k] <= s[j] {
            if s[k] < s[j] {
                k = i;
            } else {
                k += 1;
            }
            j += 1;
        }
        while i <= k {
            res.push(&s[i..i + j - k]);
            i += j - k;
        }
    }
    res
}

#[test]
fn test_factorization() {
    let s = "bbbbabracadabra".as_bytes().to_vec();
    let lf = decomposed_factorization(&s);
    assert_eq!(
        lf,
        vec!["b", "b", "b", "b", "abracad", "abr", "a"]
            .iter()
            .map(|&x| x.as_bytes())
            .collect::<Vec<&[u8]>>()
    );
    //for i in 0..lf.len() {
    //println!("{:?}", str::from_utf8(&lf[i]).unwrap());
    //}
}
