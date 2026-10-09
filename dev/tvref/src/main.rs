use std::fs;
use std::io::Write;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

const ROTATION_SEED: [u8; 32] = [
    164, 143, 161, 123, 88, 50, 61, 10, 234, 184, 161, 204, 105, 1, 20, 184, 43, 140, 200,
    117, 24, 180, 247, 84, 141, 68, 110, 161, 228, 223, 32, 242,
];

fn fisher_yates(dim: usize, rng: &mut ChaCha8Rng) -> Vec<u32> {
    let mut perm: Vec<u32> = (0..dim as u32).collect();
    for i in (1..dim).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    perm
}

struct Xs(u64);
impl Xs {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x as f64 / u64::MAX as f64) as f32 - 0.5
    }
}

fn dump_rotation(out: &str, dims: &[usize]) {
    let mut f = fs::File::create(format!("{out}/rotation.txt")).unwrap();
    for &dim in dims {
        let mut rng = ChaCha8Rng::from_seed(ROTATION_SEED);
        writeln!(f, "dim {dim}").unwrap();
        for _round in 0..2 {
            let signs: Vec<i8> = (0..dim)
                .map(|_| if rng.next_u32() & 1 == 1 { -1 } else { 1 })
                .collect();
            let perm = fisher_yates(dim, &mut rng);
            let s: String = signs.iter().map(|&v| if v < 0 { '1' } else { '0' }).collect();
            writeln!(f, "signs {s}").unwrap();
            let p: Vec<String> = perm.iter().map(|v| v.to_string()).collect();
            writeln!(f, "perm {}", p.join(",")).unwrap();
        }
    }
}

fn dump_codebook(out: &str, shapes: &[(usize, usize)]) {
    let dir = format!("{out}/codebook");
    fs::create_dir_all(&dir).unwrap();
    for &(bits, dim) in shapes {
        let (boundaries, centroids) = turbovec::expected_codebook(bits, dim);
        let mut f = fs::File::create(format!("{dir}/b{bits}_d{dim}.txt")).unwrap();
        writeln!(f, "centroids").unwrap();
        for c in &centroids {
            writeln!(f, "{:08x}", c.to_bits()).unwrap();
        }
        writeln!(f, "boundaries").unwrap();
        for b in &boundaries {
            writeln!(f, "{:08x}", b.to_bits()).unwrap();
        }
    }
}

fn dump_index(out: &str, dim: usize, bits: usize, n: usize, nq: usize, k: usize) {
    use turbovec::TurboQuantIndex;

    let mut xs = Xs(0x9E37_79B9_7F4A_7C15 ^ (dim as u64) ^ ((bits as u64) << 32));
    let vectors: Vec<f32> = (0..n * dim).map(|_| xs.next_f32()).collect();
    let queries: Vec<f32> = (0..nq * dim).map(|_| xs.next_f32()).collect();

    for calibrated in [false, true] {
        let tag = if calibrated { "cal" } else { "raw" };
        let mut index = TurboQuantIndex::new(dim, bits).unwrap();
        if calibrated {
            let sample: Vec<f32> = vectors[..(1000 * dim).min(vectors.len())].to_vec();
            index.calibrate(&sample).unwrap();
        }
        index.add(&vectors);
        let codes = index.packed_codes();
        fs::write(format!("{out}/codes_b{bits}_d{dim}_n{n}_{tag}.bin"), codes).unwrap();

        let res = index.search(&queries, k);
        let mut f = fs::File::create(format!("{out}/search_b{bits}_d{dim}_n{n}_{tag}.txt")).unwrap();
        writeln!(f, "nq {} k {}", res.nq, res.k).unwrap();
        for v in &res.scores {
            writeln!(f, "{:08x}", v.to_bits()).unwrap();
        }
        for v in &res.indices {
            writeln!(f, "{v}").unwrap();
        }
    }
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(||
        concat!(env!("CARGO_MANIFEST_DIR"), "/../out").into());
    fs::create_dir_all(&out).unwrap();
    dump_rotation(&out, &[8, 200, 768, 1536]);
    dump_codebook(&out, &[(2, 8), (2, 200), (2, 768), (2, 1536), (3, 128), (3, 768), (4, 768), (4, 1536)]);
    dump_index(&out, 768, 4, 256, 4, 10);
    dump_index(&out, 768, 2, 256, 4, 10);
    dump_index(&out, 200, 2, 200, 2, 5);
    dump_index(&out, 32, 4, 8, 2, 3);
    dump_index(&out, 32, 2, 8, 2, 3);
    dump_index(&out, 40, 3, 8, 2, 3);
    println!("wrote reference data to {out}");
}
