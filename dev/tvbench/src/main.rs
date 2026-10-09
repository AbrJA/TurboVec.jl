use std::time::Instant;

use turbovec::TurboQuantIndex;

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

fn bench(dim: usize, bits: usize, n: usize, nq: usize, k: usize) {
    let seed = 0x9E37_79B9_7F4A_7C15u64 ^ (dim as u64) ^ ((bits as u64) << 32);
    let mut xs = Xs(seed);
    let db: Vec<f32> = (0..n * dim).map(|_| xs.next_f32()).collect();
    let queries: Vec<f32> = (0..nq * dim).map(|_| xs.next_f32()).collect();

    let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
    let t = Instant::now();
    idx.add(&db);
    let add_s = t.elapsed().as_secs_f64();

    idx.prepare();
    let _ = idx.search(&queries, k);
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let res = idx.search(&queries, k);
        let e = t.elapsed().as_secs_f64();
        std::hint::black_box(res.scores.len());
        best = best.min(e);
    }
    println!(
        "rust dim={dim} bits={bits} n={n} threads={} add={add_s:.3}s search={:.3}ms/q",
        rayon::current_num_threads(),
        best / nq as f64 * 1000.0
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shape: Vec<(usize, usize, usize)> = if args.len() > 1 {
        vec![(args[1].parse().unwrap(), args[2].parse().unwrap(), args[3].parse().unwrap())]
    } else {
        vec![(768, 4, 100_000), (768, 2, 100_000), (1536, 4, 50_000), (1536, 2, 50_000)]
    };
    for (dim, bits, n) in shape {
        bench(dim, bits, n, 100, 64);
    }
}
