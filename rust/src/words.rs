// Fibonacci words
// 0 : b
// 1 : a
// 2 : ab
// 3 : aba
// 4:  abaab
// i : f(i-1) f(i-2)

pub fn fibonacci(i: usize) -> Vec<u8> {
    let mut res = Vec::new();
    let mut i = i;
    if i == 0 {
        res.push(b'b');
    } else if i == 1 {
        res.push(b'a');
    } else {
        // i >= 2
        let mut p1 = 2;
        let mut p2 = 1;
        res.push(b'a');
        res.push(b'b');
        while i >= 3 {
            for j in 0..p2 {
                res.push(res[j]);
            }
            i = i - 1;
            let n = p1 + p2;
            p2 = p1;
            p1 = n;
        }
    }
    res
}

pub fn fibonacci_plus(i: usize) -> Vec<u8> {
    let mut res = fibonacci(i);
    res.push(if i % 2 == 0 { b'b' } else { b'a' });
    res
}

// construct ith Thue-Morse word
// 0: a, 1: ab, 2: abba
pub fn thue_morse(i: usize) -> Vec<u8> {
    let mut res = Vec::new();
    res.push(b'a');
    let mut i = i;
    while i > 0 {
        let l = res.len();
        for j in 0..l {
            if res[j] == b'a' {
                res.push(b'b');
            } else {
                res.push(b'a');
            }
        }
        i = i - 1;
    }
    res
}

pub fn period_doubling(i: usize) -> Vec<u8> {
    let mut res = Vec::new();
    res.push(b'a');
    let mut i = i;
    if i > 0 {
        res.push(b'b');
        i -= 1;
    }
    while i > 0 {
        let l = res.len();
        for j in 0..l / 2 {
            if res[l / 2 + j] == b'a' {
                res.push(b'a');
                res.push(b'b');
            } else {
                res.push(b'a');
                res.push(b'a');
            }
        }
        i -= 1;
    }
    res
}

// generate all strings of certain alphabet and length
#[allow(dead_code)]
fn genall_dfs_aux(s: &mut Vec<u8>, alphabet: &[u8], len: usize) {
    if s.len() == len {
        println!("{}", String::from_utf8(s.to_vec()).unwrap());
    } else {
        for c in alphabet {
            s.push(*c);
            genall_dfs_aux(s, alphabet, len);
            s.pop();
        }
    }
}

#[allow(dead_code)]
pub fn genall_dfs(alphabet: &[u8], len: usize) {
    let mut s = Vec::new();
    genall_dfs_aux(&mut s, &alphabet, len);
}

#[test]
fn test_fibonacci() {
    assert_eq!(fibonacci(0), "b".as_bytes().to_vec());
    assert_eq!(fibonacci(1), "a".as_bytes().to_vec());
    assert_eq!(fibonacci(2), "ab".as_bytes().to_vec());
    assert_eq!(fibonacci(3), "aba".as_bytes().to_vec());
    assert_eq!(
        fibonacci(10),
        "abaababaabaababaababaabaababaabaababaababaabaababaababaabaababaabaababaababaabaababaabaab"
            .as_bytes()
            .to_vec()
    );
    assert_eq!(
        fibonacci_plus(8),
        "abaababaabaababaababaabaababaabaabb".as_bytes().to_vec()
    )
}
#[test]
fn test_thue_mores() {
    assert_eq!(
        thue_morse(5),
        "abbabaabbaababbabaababbaabbabaab".as_bytes().to_vec()
    );
}

#[test]
fn test_period_doubling() {
    assert_eq!(period_doubling(3), "abaaabab".as_bytes().to_vec());
    assert_eq!(period_doubling(4), "abaaabababaaabaa".as_bytes().to_vec())
}
