const MULTIPLIER: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

fn coerce_to_words(seed: u128) -> Vec<u32> {
    if seed == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    let mut value = seed;
    while value > 0 {
        words.push((value & 0xffff_ffff) as u32);
        value >>= 32;
    }
    words
}

fn hashmix(value: u32, constant: &mut u32) -> u32 {
    let mut mixed = value ^ *constant;
    *constant = constant.wrapping_mul(MULT_A);
    mixed = mixed.wrapping_mul(*constant);
    mixed ^ (mixed >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

pub fn seed_sequence_pool(entropy: u128) -> [u32; POOL_SIZE] {
    let words = coerce_to_words(entropy);
    let mut constant = INIT_A;
    let mut pool = [0u32; POOL_SIZE];
    for (index, slot) in pool.iter_mut().enumerate() {
        let source = words.get(index).copied().unwrap_or(0);
        *slot = hashmix(source, &mut constant);
    }
    for source in 0..POOL_SIZE {
        for destination in 0..POOL_SIZE {
            if source != destination {
                let mixed = hashmix(pool[source], &mut constant);
                pool[destination] = mix(pool[destination], mixed);
            }
        }
    }
    for source in POOL_SIZE..words.len() {
        for destination in 0..POOL_SIZE {
            let mixed = hashmix(words[source], &mut constant);
            pool[destination] = mix(pool[destination], mixed);
        }
    }
    pool
}

pub fn generate_state(pool: &[u32; POOL_SIZE], words: usize) -> Vec<u32> {
    let mut constant = INIT_B;
    let mut state = Vec::with_capacity(words);
    for index in 0..words {
        let mut value = pool[index % POOL_SIZE];
        value ^= constant;
        constant = constant.wrapping_mul(MULT_B);
        value = value.wrapping_mul(constant);
        state.push(value ^ (value >> XSHIFT));
    }
    state
}

#[derive(Clone)]
pub struct Pcg64 {
    state: u128,
    increment: u128,
    cached: Option<u32>,
}

impl Pcg64 {
    pub fn from_seed(seed: u128) -> Self {
        let pool = seed_sequence_pool(seed);
        let words = generate_state(&pool, 8);
        let initial = (u128::from(words[1]) << 96)
            | (u128::from(words[0]) << 64)
            | (u128::from(words[3]) << 32)
            | u128::from(words[2]);
        let sequence = (u128::from(words[5]) << 96)
            | (u128::from(words[4]) << 64)
            | (u128::from(words[7]) << 32)
            | u128::from(words[6]);
        let mut generator = Self {
            state: 0,
            increment: (sequence << 1) | 1,
            cached: None,
        };
        generator.step();
        generator.state = generator.state.wrapping_add(initial);
        generator.step();
        generator
    }

    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment);
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let state = self.state;
        let value = ((state >> 64) as u64) ^ (state as u64);
        value.rotate_right((state >> 122) as u32)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if let Some(value) = self.cached.take() {
            return value;
        }
        let next = self.next_u64();
        self.cached = Some((next >> 32) as u32);
        next as u32
    }

    pub fn bounded_u32(&mut self, range: u32) -> u32 {
        let exclusive = u64::from(range) + 1;
        let mut product = u64::from(self.next_u32()) * exclusive;
        let mut leftover = product as u32;
        if u64::from(leftover) < exclusive {
            let threshold = ((u32::MAX - range) as u64 % exclusive) as u32;
            while leftover < threshold {
                product = u64::from(self.next_u32()) * exclusive;
                leftover = product as u32;
            }
        }
        (product >> 32) as u32
    }
}

pub fn head_from_seed(seed: u128, heads: u32) -> usize {
    Pcg64::from_seed(seed).bounded_u32(heads - 1) as usize
}
