const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

#[derive(Clone)]
pub struct Mt19937 {
    key: [u32; N],
    pos: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut key = [0u32; N];
        let mut value = seed;
        for (position, slot) in key.iter_mut().enumerate() {
            *slot = value;
            value = 1812433253u32
                .wrapping_mul(value ^ (value >> 30))
                .wrapping_add(position as u32 + 1);
        }
        Self { key, pos: N }
    }

    fn twist(&mut self) {
        for index in 0..N - M {
            let y = (self.key[index] & UPPER_MASK) | (self.key[index + 1] & LOWER_MASK);
            self.key[index] = self.key[index + M] ^ (y >> 1) ^ (y & 1).wrapping_neg() & MATRIX_A;
        }
        for index in N - M..N - 1 {
            let y = (self.key[index] & UPPER_MASK) | (self.key[index + 1] & LOWER_MASK);
            self.key[index] =
                self.key[index + M - N] ^ (y >> 1) ^ (y & 1).wrapping_neg() & MATRIX_A;
        }
        let y = (self.key[N - 1] & UPPER_MASK) | (self.key[0] & LOWER_MASK);
        self.key[N - 1] = self.key[M - 1] ^ (y >> 1) ^ (y & 1).wrapping_neg() & MATRIX_A;
        self.pos = 0;
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= N {
            self.twist();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}
