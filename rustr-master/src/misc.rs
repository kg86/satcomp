// border_array[i] is the longest border of s[0..i]
// border_array[0] = 0
pub fn border_array(s: &[u8]) -> Vec<i32> {
  let mut res = Vec::new();
  res.push(0);
  let mut j: i32 = 0;
  for i in 1..s.len() {
    while s[i] != s[j as usize] {
      if j == 0 {
        j -= 1;
        break;
      }
      j = res[j as usize];
    }
    j += 1;
    res.push(j);
  }
  res
}

#[test]
fn test_border_array() {
  let s = "abracadabra".as_bytes().to_vec();
  let b = border_array(&s);
  assert_eq!(b, vec![0, 0, 0, 1, 0, 1, 0, 1, 2, 3, 4]);
}
