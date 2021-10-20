// functions for run length encodings

pub fn runlength(s: &[u8]) -> i32 {
    let mut res = if s.len() == 0 { 0 } else { 1 };
    for i in 1..s.len() {
        if s[i - 1] != s[i] {
            res += 1;
        }
    }
    res
}

pub fn encode(s: &[u8]) -> Vec<(u8, i32)> {
    let mut res = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let mut exp = 1;
        let c = s[i];
        i += 1;
        while i < s.len() && s[i] == c {
            i += 1;
            exp += 1;
        }
        res.push((c, exp));
    }
    res
}

#[test]
fn test_rle() {
    assert_eq!(runlength(&vec![1, 1, 2, 1, 1, 3, 3, 3, 1]), 5);
    assert_eq!(runlength(&vec![1, 1, 2, 1, 1, 3, 3, 3]), 4);
}

#[test]
fn test_rle_encode() {
    assert_eq!(
        encode(&vec![1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2]),
        vec![(1, 1), (2, 4), (3, 3), (2, 3)]
    );
    assert_eq!(
        encode(&vec![1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 0]),
        vec![(1, 1), (2, 4), (3, 3), (2, 3), (0, 1)]
    );
}
