//! Gradient regions: after the image is reduced to its palette, a gradient is cut into bands of
//! flat colour. Here neighbouring regions are merged again where nothing but the palette parts
//! them (the pixels change little across their shared edge) and the merged area's own pixels are
//! one gradient, either
//!
//! - **linear**: each colour channel a smooth function (a parabola, so a highlight across the
//!   middle of a shape is one gradient too) of the distance along one direction, or
//! - **radial**: each channel a function of the distance from one centre, so the bands of colour
//!   are arcs (the shading of a curved shape, such as a feather lit along its edge).
//!
//! Each merged area becomes one path filled with that gradient, its stops following its colours.
//!
//! Regions are merged cheapest first (the least added squared error), as long as the merged area
//! stays within a tolerance of its gradient. The fits are least squares over polynomials of the
//! position of degree two at most, computed from running sums of the pixels' position monomials
//! (to degree four) and colours, so trying a merge costs the same whatever the regions' size.
//! A merged area whose drawn gradient turns out further from its pixels than the tolerance allows
//! is left as the regions it was made of.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use kurbo::{Point, Vec2};

use crate::quantize::{Quantized, TRANSPARENT};
use crate::{Raster, TracedGradient};

/// The largest root-mean-square difference (levels of a channel) between a merged area's pixels and
/// its gradient.
/// (Measured on real logos: 5 leaves curved shading in bands, 12 merges it.)
const TOLERANCE: f64 = 12.0;
/// The largest mean step (levels, the channel that changes most) between the pixels on either side
/// of the edge two regions share for them to merge: more is an edge in the picture.
const MAX_STEP: f64 = 12.0;
/// How much further than the tolerance the drawn gradient of a merged area may be from its pixels.
const DRAWN_SLACK: f64 = 1.5;
/// How far (levels) the stops' ramps may be from the colours along the gradient, and the most stops.
const STOP_TOLERANCE: f64 = 3.0;
const MAX_STOPS: usize = 8;
/// The least change (levels) across an area for it to be a gradient rather than one flat colour.
const MIN_CHANGE: f64 = 8.0;
/// The fewest pixels a merged area has to be a gradient.
const MIN_PIXELS: usize = 64;
/// The farthest a radial gradient's centre may be (in half image sizes): beyond, its arcs are
/// straight lines and the linear gradient serves.
const FAR_CENTRE: f64 = 40.0;

/// Index of monomial x^a·y^b (degree d = a + b ≤ 4) in [`Moments::m`].
const fn mono(a: usize, b: usize) -> usize {
    let d = a + b;
    d * (d + 1) / 2 + b
}

/// The exponents of each of the six monomials of degree two at most, in [`mono`] order.
const POWERS: [(usize, usize); 6] = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)];

/// A polynomial of the position of degree two at most: coefficients of 1, x, y, x², xy, y².
type Poly = [f64; 6];

/// The product of two polynomials of degree one at most.
fn mul1(p: &Poly, q: &Poly) -> Poly {
    [p[0] * q[0], p[0] * q[1] + p[1] * q[0], p[0] * q[2] + p[2] * q[0], p[1] * q[1], p[1] * q[2] + p[2] * q[1], p[2] * q[2]]
}

/// A colour model: its basis functions of the position.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Model {
    /// Along direction `u`.
    Linear { u: Vec2 },
    /// About `centre`.
    Radial { centre: Point },
}

/// Running sums over a region's pixels (positions scaled to about −1..1): every monomial of the
/// position up to degree 4, and per channel the colour times every monomial up to degree 2 and
/// the colour squared.
#[derive(Clone, Copy, Debug)]
struct Moments {
    m: [f64; 15],
    c: [[f64; 7]; 3],
}

impl Moments {
    const ZERO: Moments = Moments { m: [0.0; 15], c: [[0.0; 7]; 3] };

    fn add(&mut self, x: f64, y: f64, rgb: [f64; 3]) {
        let xp = [1.0, x, x * x, x * x * x, x * x * x * x];
        let yp = [1.0, y, y * y, y * y * y, y * y * y * y];
        for d in 0..=4 {
            for b in 0..=d {
                if let (Some(slot), Some(xa), Some(yb)) = (self.m.get_mut(mono(d - b, b)), xp.get(d - b), yp.get(b)) {
                    *slot += xa * yb;
                }
            }
        }
        for (ch, &v) in self.c.iter_mut().zip(&rgb) {
            for (k, &(a, b)) in POWERS.iter().enumerate() {
                if let (Some(slot), Some(xa), Some(yb)) = (ch.get_mut(k), xp.get(a), yp.get(b)) {
                    *slot += v * xa * yb;
                }
            }
            ch[6] += v * v;
        }
    }

    fn merged(&self, o: &Moments) -> Moments {
        let mut out = *self;
        out.m.iter_mut().zip(o.m).for_each(|(a, b)| *a += b);
        for (a, b) in out.c.iter_mut().zip(o.c) {
            a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
        }
        out
    }

    fn n(&self) -> f64 {
        self.m[0]
    }

    fn mean(&self) -> Point {
        let n = self.n().max(1e-300);
        Point::new(self.m[mono(1, 0)] / n, self.m[mono(0, 1)] / n)
    }

    /// Σ p·q over the pixels.
    fn sum2(&self, p: &Poly, q: &Poly) -> f64 {
        let mut s = 0.0;
        for (i, &(a, b)) in POWERS.iter().enumerate() {
            for (j, &(c, d)) in POWERS.iter().enumerate() {
                let k = p[i] * q[j];
                if k != 0.0 {
                    s += k * self.m.get(mono(a + c, b + d)).copied().unwrap_or(0.0);
                }
            }
        }
        s
    }

    /// Σ colour·p over the pixels, for each channel.
    fn sum_c(&self, p: &Poly) -> [f64; 3] {
        self.c.map(|ch| (0..6).map(|i| p[i] * ch[i]).sum())
    }

    /// `p` made zero-mean and unit-variance over the pixels (`None` when it is constant on them).
    fn standardised(&self, p: Poly) -> Option<Poly> {
        let n = self.n();
        let one: Poly = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mean = self.sum2(&p, &one) / n;
        let var = self.sum2(&p, &p) / n - mean * mean;
        if !(var.is_finite() && var > 1e-14) {
            return None;
        }
        let sd = var.sqrt();
        let mut q = p.map(|v| v / sd);
        q[0] -= mean / sd;
        Some(q)
    }

    /// The least-squares fit of each channel to `basis` (the constant is always part of it): the
    /// coefficients per channel, and the sum of squared differences over all three.
    fn fit(&self, basis: &[Poly]) -> Option<(Vec<[f64; 3]>, f64)> {
        let mut all: Vec<Poly> = vec![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        all.extend_from_slice(basis);
        let k = all.len();
        let g: Vec<Vec<f64>> = all.iter().map(|p| all.iter().map(|q| self.sum2(p, q)).collect()).collect();
        let r: Vec<[f64; 3]> = all.iter().map(|p| self.sum_c(p)).collect();
        let beta = solve(g, r.clone())?;
        let mut sse = 0.0;
        for ch in 0..3 {
            let explained: f64 = (0..k).map(|i| beta.get(i).map_or(0.0, |b| b[ch]) * r.get(i).map_or(0.0, |r| r[ch])).sum();
            sse += self.c[ch][6] - explained;
        }
        sse.is_finite().then_some((beta, sse.max(0.0)))
    }

    /// The sum of squared differences from the region's mean colour.
    fn flat_sse(&self) -> f64 {
        let n = self.n().max(1e-300);
        self.c.iter().map(|ch| ch[6] - ch[0] * ch[0] / n).sum::<f64>().max(0.0)
    }

    /// The direction the colours change along: the longer axis of the three channels' plane
    /// slopes together, or of the pixels' own spread when the planes cannot be fitted.
    fn direction(&self) -> Vec2 {
        let n = self.n().max(1.0);
        let c = self.mean();
        let sxx = self.m[mono(2, 0)] - n * c.x * c.x;
        let sxy = self.m[mono(1, 1)] - n * c.x * c.y;
        let syy = self.m[mono(0, 2)] - n * c.y * c.y;
        let axis = |a: f64, b: f64, c: f64| {
            let angle = 0.5 * (2.0 * b).atan2(a - c);
            Vec2::new(angle.cos(), angle.sin())
        };
        let Some((beta, _)) = self.fit(&[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]) else { return axis(sxx, sxy, syy) };
        let (Some(bx), Some(by)) = (beta.get(1), beta.get(2)) else { return axis(sxx, sxy, syy) };
        let (a, b, cc) = (0..3).fold((0.0, 0.0, 0.0), |m, k| (m.0 + bx[k] * bx[k], m.1 + bx[k] * by[k], m.2 + by[k] * by[k]));
        if a + cc <= 0.0 { axis(sxx, sxy, syy) } else { axis(a, b, cc) }
    }

    /// The centre the colours change about, if they change in rings about one: from each channel's
    /// best fit `a + b·p + β·|p|²` (rings about −b / 2β), the centre all three agree on best.
    fn centre(&self) -> Option<Point> {
        let m = self.mean();
        let (dx, dy) = ([-m.x, 1.0, 0.0, 0.0, 0.0, 0.0], [-m.y, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let ring = {
            let (xx, yy) = (mul1(&dx, &dx), mul1(&dy, &dy));
            std::array::from_fn::<f64, 6, _>(|i| xx[i] + yy[i])
        };
        let ring = self.standardised(ring)?;
        let (sx, sy) = (self.standardised(dx)?, self.standardised(dy)?);
        let (beta, _) = self.fit(&[sx, sy, ring])?;
        // Back to the position: channel = … + bx·(x − mx)/sdx + by·(y − my)/sdy + β·(|p − m|² − …)/sdr.
        let (bx, by, br) = (beta.get(1)?, beta.get(2)?, beta.get(3)?);
        let (kx, ky, kr) = (sx[1], sy[2], ring[3]);
        let (mut num, mut den) = (Vec2::ZERO, 0.0);
        for ch in 0..3 {
            let b = Vec2::new(bx[ch] * kx, by[ch] * ky);
            let q = br[ch] * kr;
            num += b * q;
            den += q * q;
        }
        if !(den.is_finite() && den > 1e-12) {
            return None;
        }
        let c = m - num / (2.0 * den);
        (c.to_vec2().hypot() < FAR_CENTRE && c.x.is_finite() && c.y.is_finite()).then_some(c)
    }

    /// The best model's sum of squared differences, and the model.
    fn best(&self) -> (f64, Option<Model>) {
        let flat = self.flat_sse();
        if self.n() < 8.0 {
            return (flat, None);
        }
        let mut best = (flat, None);
        let u = self.direction();
        let s = self.standardised([0.0, u.x, u.y, 0.0, 0.0, 0.0]);
        if let Some(s) = s {
            let s2 = self.standardised(mul1(&s, &s));
            let basis: Vec<Poly> = std::iter::once(s).chain(s2).collect();
            if let Some((_, e)) = self.fit(&basis)
                && e < best.0
            {
                best = (e, Some(Model::Linear { u }));
            }
        }
        if let Some(c) = self.centre() {
            let (dx, dy) = ([-c.x, 1.0, 0.0, 0.0, 0.0, 0.0], [-c.y, 0.0, 1.0, 0.0, 0.0, 0.0]);
            let (xx, yy) = (mul1(&dx, &dx), mul1(&dy, &dy));
            let r2: Poly = std::array::from_fn(|i| xx[i] + yy[i]);
            if let Some(r2) = self.standardised(r2)
                && let Some((_, e)) = self.fit(&[r2])
                && e < best.0
            {
                best = (e, Some(Model::Radial { centre: c }));
            }
        }
        best
    }
}

/// Solves `g · x = r` (three right-hand sides) by Gaussian elimination with partial pivoting.
fn solve(mut g: Vec<Vec<f64>>, mut r: Vec<[f64; 3]>) -> Option<Vec<[f64; 3]>> {
    let k = g.len();
    let at = |g: &[Vec<f64>], i: usize, j: usize| g.get(i).and_then(|row| row.get(j)).copied().unwrap_or(0.0);
    let scale = (0..k).map(|i| at(&g, i, i).abs()).fold(0.0, f64::max);
    for col in 0..k {
        let pivot = (col..k).max_by(|&a, &b| at(&g, a, col).abs().total_cmp(&at(&g, b, col).abs()))?;
        let big = at(&g, pivot, col).abs();
        if big.is_nan() || big <= 1e-10 * scale {
            return None;
        }
        g.swap(col, pivot);
        r.swap(col, pivot);
        let (top, bottom) = g.split_at_mut(col + 1);
        let (rtop, rbottom) = r.split_at_mut(col + 1);
        let (prow, pr) = (top.last()?, rtop.last()?);
        let p = *prow.get(col)?;
        for (row, rr) in bottom.iter_mut().zip(rbottom.iter_mut()) {
            let f = row.get(col)? / p;
            for (a, b) in row.iter_mut().zip(prow).skip(col) {
                *a -= f * b;
            }
            for (a, b) in rr.iter_mut().zip(pr) {
                *a -= f * b;
            }
        }
    }
    let mut x = vec![[0.0; 3]; k];
    for row in (0..k).rev() {
        let d = at(&g, row, row);
        let rr = *r.get(row)?;
        let mut v = [0.0; 3];
        for (ch, out) in v.iter_mut().enumerate() {
            let s: f64 = (row + 1..k).map(|j| at(&g, row, j) * x.get(j).map_or(0.0, |x| x[ch])).sum();
            *out = (rr[ch] - s) / d;
        }
        *x.get_mut(row)? = v;
    }
    x.iter().all(|v| v.iter().all(|c| c.is_finite())).then_some(x)
}

/// The edge two regions share: how many neighbouring pixel pairs it has and the sum of their
/// steps.
#[derive(Clone, Copy, Default)]
struct Edge {
    pairs: f64,
    steps: f64,
}

impl Edge {
    fn add(&mut self, o: Edge) {
        self.pairs += o.pairs;
        self.steps += o.steps;
    }

    /// Nothing but the palette parts the two sides: the pixels change little across it.
    fn soft(&self) -> bool {
        self.pairs > 0.0 && self.steps / self.pairs <= MAX_STEP
    }
}

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while let Some(&p) = parent.get(i as usize) {
        if p == i {
            break;
        }
        let gp = parent.get(p as usize).copied().unwrap_or(p);
        if let Some(slot) = parent.get_mut(i as usize) {
            *slot = gp;
        }
        i = gp;
    }
    i
}

fn rgb_of(img: &Raster, i: usize) -> [f64; 3] {
    img.rgba.get(i * 4..i * 4 + 3).map_or([0.0; 3], |p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
}

/// The pixels beside pixel `i` of a `w` × `h` image.
fn beside(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % w, i / w);
    [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten()
}

/// The image's connected regions of one label (4-connected, as traced) that may be merged.
struct Regions {
    w: usize,
    h: usize,
    /// Positions are scaled by this about the image's middle for the moments.
    scale: f64,
    /// Each pixel's region, or `u32::MAX`.
    of: Vec<u32>,
    moments: Vec<Moments>,
}

impl Regions {
    fn find(img: &Raster, q: &Quantized, gradients: &[Option<TracedGradient>]) -> Regions {
        let (w, h) = (img.width as usize, img.height as usize);
        let mergeable = |l: u16| l != TRANSPARENT && gradients.get(l as usize).is_some_and(Option::is_none);
        let scale = (w.max(h) as f64 / 2.0).max(1.0);
        let mut r = Regions { w, h, scale, of: vec![u32::MAX; w * h], moments: vec![] };
        let mut stack = vec![];
        for seed in 0..w * h {
            let l = q.labels.get(seed).copied().unwrap_or(TRANSPARENT);
            if r.of.get(seed).copied() != Some(u32::MAX) || !mergeable(l) {
                continue;
            }
            let id = r.moments.len() as u32;
            let mut mo = Moments::ZERO;
            r.of[seed] = id;
            stack.push(seed);
            while let Some(i) = stack.pop() {
                let p = r.scaled(i);
                mo.add(p.x, p.y, rgb_of(img, i));
                for j in beside(i, w, h) {
                    if r.of.get(j).copied() == Some(u32::MAX) && q.labels.get(j).copied() == Some(l) {
                        r.of[j] = id;
                        stack.push(j);
                    }
                }
            }
            r.moments.push(mo);
        }
        r
    }

    /// Pixel `i`'s middle, scaled.
    fn scaled(&self, i: usize) -> Point {
        let (w, h) = (self.w as f64, self.h as f64);
        Point::new(((i % self.w) as f64 + 0.5 - w / 2.0) / self.scale, ((i / self.w) as f64 + 0.5 - h / 2.0) / self.scale)
    }

    /// A scaled point back in pixels.
    fn unscaled(&self, p: Point) -> Point {
        Point::new(p.x * self.scale + self.w as f64 / 2.0, p.y * self.scale + self.h as f64 / 2.0)
    }

    /// The edges between regions, with the steps across them.
    fn edges(&self, img: &Raster) -> Vec<HashMap<u32, Edge>> {
        let mut edges: Vec<HashMap<u32, Edge>> = vec![HashMap::new(); self.moments.len()];
        for i in 0..self.w * self.h {
            let a = self.of.get(i).copied().unwrap_or(u32::MAX);
            if a == u32::MAX {
                continue;
            }
            let right = (i % self.w + 1 < self.w).then(|| i + 1);
            let down = (i / self.w + 1 < self.h).then(|| i + self.w);
            for j in [right, down].into_iter().flatten() {
                let b = self.of.get(j).copied().unwrap_or(u32::MAX);
                if b == u32::MAX || b == a {
                    continue;
                }
                let (ca, cb) = (rgb_of(img, i), rgb_of(img, j));
                let steps = (0..3).map(|k| (ca[k] - cb[k]).abs()).fold(0.0, f64::max);
                for (x, y) in [(a, b), (b, a)] {
                    if let Some(map) = edges.get_mut(x as usize) {
                        map.entry(y).or_default().add(Edge { pairs: 1.0, steps });
                    }
                }
            }
        }
        edges
    }
}

/// A candidate merge, cheapest first: (added error, a, b, a's version, b's version).
type Candidate = Reverse<(u64, u32, u32, u32, u32)>;

/// The merging of regions: each merged area is kept at its first region (its root).
struct Merger {
    moments: Vec<Moments>,
    sse: Vec<f64>,
    version: Vec<u32>,
    parent: Vec<u32>,
    members: Vec<u32>,
    edges: Vec<HashMap<u32, Edge>>,
    heap: BinaryHeap<Candidate>,
}

impl Merger {
    fn new(moments: Vec<Moments>, edges: Vec<HashMap<u32, Edge>>) -> Merger {
        let count = moments.len();
        let sse = moments.iter().map(|m| m.best().0).collect();
        let mut m = Merger {
            moments,
            sse,
            version: vec![0; count],
            parent: (0..count as u32).collect(),
            members: vec![1; count],
            edges,
            heap: BinaryHeap::new(),
        };
        for a in 0..count as u32 {
            m.offer(a, |b| a < b);
        }
        m
    }

    /// The error merging `a` and `b` adds, and the merged error, if the merged area is still
    /// within the tolerance of its gradient.
    fn cost(&self, a: u32, b: u32) -> Option<(f64, f64)> {
        let m = self.moments.get(a as usize)?.merged(self.moments.get(b as usize)?);
        let e = m.best().0;
        let rms = (e / (3.0 * m.n()).max(1.0)).sqrt();
        let parts = self.sse.get(a as usize)? + self.sse.get(b as usize)?;
        (rms <= TOLERANCE).then(|| ((e - parts).max(0.0), e))
    }

    /// Queues the merges of `a` with its neighbours that `with` picks.
    fn offer(&mut self, a: u32, with: impl Fn(u32) -> bool) {
        let next: Vec<u32> =
            self.edges.get(a as usize).map_or(vec![], |map| map.iter().filter(|(c, e)| with(**c) && e.soft()).map(|(&c, _)| c).collect());
        let version = |m: &Merger, r: u32| m.version.get(r as usize).copied().unwrap_or(0);
        for c in next {
            if let Some((cost, _)) = self.cost(a, c) {
                let (x, y) = (a.min(c), a.max(c));
                self.heap.push(Reverse((cost.to_bits(), x, y, version(self, x), version(self, y))));
            }
        }
    }

    /// Merges `b` into `a` (both roots), its error `e`.
    fn join(&mut self, a: u32, b: u32, e: f64) -> Option<()> {
        let mb = *self.moments.get(b as usize)?;
        let nb = *self.members.get(b as usize)?;
        let ma = self.moments.get_mut(a as usize)?;
        *ma = ma.merged(&mb);
        *self.sse.get_mut(a as usize)? = e;
        *self.parent.get_mut(b as usize)? = a;
        *self.members.get_mut(a as usize)? += nb;
        *self.version.get_mut(a as usize)? += 1;
        let from_b = self.edges.get_mut(b as usize).map(std::mem::take).unwrap_or_default();
        for (c, eb) in from_b {
            if c == a {
                continue;
            }
            if let Some(map) = self.edges.get_mut(c as usize) {
                map.remove(&b);
                map.entry(a).or_default().add(eb);
            }
            if let Some(map) = self.edges.get_mut(a as usize) {
                map.entry(c).or_default().add(eb);
            }
        }
        self.edges.get_mut(a as usize)?.remove(&b);
        Some(())
    }

    /// Merges, cheapest first, while any merge keeps within the tolerance.
    fn run(&mut self) {
        while let Some(Reverse((_, a, b, va, vb))) = self.heap.pop() {
            let current = self.version.get(a as usize) == Some(&va) && self.version.get(b as usize) == Some(&vb);
            if !current || find(&mut self.parent, a) != a || find(&mut self.parent, b) != b {
                continue;
            }
            let Some((_, e)) = self.cost(a, b) else { continue };
            if self.join(a, b, e).is_some() {
                self.offer(a, |_| true);
            }
        }
    }
}

/// Merges the regions of `q` (labels after noise removal; `gradients` the gradient of each palette
/// entry, if any, those left alone) that together are one gradient, as described above. Each
/// merged area gets a palette entry of its own: its gradient, or its mean colour where it is flat.
pub(crate) fn merge_gradients(img: &Raster, q: &mut Quantized, gradients: &mut Vec<Option<TracedGradient>>) {
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || q.labels.len() != w * h || img.rgba.len() != w * h * 4 {
        return;
    }
    let regions = Regions::find(img, q, gradients);
    let count = regions.moments.len();
    if count < 2 || count > u32::MAX as usize / 2 {
        return;
    }
    let mut merger = Merger::new(regions.moments.clone(), regions.edges(img));
    merger.run();
    // Each merged area's pixels.
    let mut pixels: HashMap<u32, Vec<usize>> = HashMap::new();
    let roots: Vec<u32> = (0..count as u32).map(|r| find(&mut merger.parent, r)).collect();
    for (i, &c) in regions.of.iter().enumerate() {
        if let Some(&r) = roots.get(c as usize)
            && merger.members.get(r as usize).is_some_and(|&m| m >= 2)
        {
            pixels.entry(r).or_default().push(i);
        }
    }
    let mut merged: Vec<u32> = pixels.keys().copied().collect();
    merged.sort_unstable();
    for r in merged {
        let (Some(px), Some(mo)) = (pixels.get(&r), merger.moments.get(r as usize)) else { continue };
        let Ok(label) = u16::try_from(q.palette.len()) else { break };
        if label == TRANSPARENT {
            break;
        }
        // Drawn too far from its pixels: left as the regions it was made of.
        let Some((color, g)) = paint(img, &regions, px, mo) else { continue };
        for &i in px {
            if let Some(l) = q.labels.get_mut(i) {
                *l = label;
            }
        }
        q.palette.push(color);
        gradients.push(g);
    }
}

/// How merged area `px` (its moments `mo`) is painted: its colour and gradient, if it is drawn
/// within the tolerance of its pixels.
fn paint(img: &Raster, regions: &Regions, px: &[usize], mo: &Moments) -> Option<([u8; 3], Option<TracedGradient>)> {
    let w = regions.w;
    let n = px.len() as f64;
    let mean = px.iter().fold([0.0; 3], |m, &i| {
        let c = rgb_of(img, i);
        [m[0] + c[0], m[1] + c[1], m[2] + c[2]]
    });
    let mean = mean.map(|v| v / n);
    let off = |paint: &dyn Fn(usize) -> [f64; 3]| {
        let e: f64 = px
            .iter()
            .map(|&i| {
                let (a, b) = (rgb_of(img, i), paint(i));
                (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>()
            })
            .sum();
        (e / (3.0 * n)).sqrt()
    };
    let middle = |i: usize| Point::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
    // The drawn gradient of the best model, and how far it is from the pixels.
    let g = (px.len() >= MIN_PIXELS)
        .then(|| match mo.best().1 {
            Some(Model::Linear { u }) => fit(img, px, Shape::Linear(u)),
            Some(Model::Radial { centre }) => {
                let c = refine_centre(img, px, regions.unscaled(centre));
                fit(img, px, Shape::Radial(c)).or_else(|| fit(img, px, Shape::Linear(mo.direction())))
            }
            None => None,
        })
        .flatten();
    let flat_off = off(&|_| mean);
    let (g, drawn_off) = match g {
        Some(g) => {
            let e = off(&|i| g.color_at(middle(i)));
            if e < flat_off { (Some(g), e) } else { (None, flat_off) }
        }
        None => (None, flat_off),
    };
    if drawn_off > TOLERANCE * DRAWN_SLACK {
        return None;
    }
    let to8 = |c: [f64; 3]| c.map(|v| v.round().clamp(0.0, 255.0) as u8);
    let color = g.as_ref().map_or(to8(mean), |g| to8(g.color_at(g.start.midpoint(g.end))));
    Some((color, g))
}

/// How well the colours of pixels `px` of `img` are a function of their distance from `c`: the
/// sum of squared differences from the mean colour of their ring (rings a pixel wide at most,
/// 48 at most), over at most a few thousand of the pixels.
fn ring_error(img: &Raster, px: &[usize], c: Point) -> f64 {
    let w = img.width as usize;
    let every = (px.len() / 4096).max(1);
    let sample = || px.iter().step_by(every).copied();
    let r = |i: usize| (Point::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5) - c).hypot();
    let (lo, hi) = sample().fold((f64::MAX, f64::MIN), |m, i| (m.0.min(r(i)), m.1.max(r(i))));
    let bins = ((hi - lo).ceil() as usize).clamp(1, 48);
    let mut sums = vec![([0.0; 3], [0.0; 3], 0.0f64); bins];
    for i in sample() {
        let b = (((r(i) - lo) / (hi - lo).max(1e-9)) * bins as f64).floor().clamp(0.0, (bins - 1) as f64) as usize;
        if let (Some(s), Some(p)) = (sums.get_mut(b), img.rgba.get(i * 4..i * 4 + 3)) {
            for ((sum, sq), &v) in s.0.iter_mut().zip(s.1.iter_mut()).zip(p) {
                let v = f64::from(v);
                *sum += v;
                *sq += v * v;
            }
            s.2 += 1.0;
        }
    }
    sums.iter().filter(|s| s.2 > 0.0).map(|s| (0..3).map(|k| s.1[k] - s.0[k] * s.0[k] / s.2).sum::<f64>()).sum()
}

/// The centre near `c` about which the colours of `px` best form rings (a pattern search).
fn refine_centre(img: &Raster, px: &[usize], c: Point) -> Point {
    let w = img.width as usize;
    let n = px.len().max(1) as f64;
    let mean = px.iter().fold(Vec2::ZERO, |m, &i| m + Vec2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5)) / n;
    // The fit that estimated `c` puts it too far out where the colours change with the distance
    // rather than its square: first the best of a few places on the way in from it.
    let (mut best, mut err) = [0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]
        .into_iter()
        .map(|k| {
            let p = (mean + (c.to_vec2() - mean) * k).to_point();
            (p, ring_error(img, px, p))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((c, f64::MAX));
    let mut step = ((best.to_vec2() - mean).hypot() * 0.1).max(1.0);
    for _ in 0..60 {
        if step < 0.25 {
            break;
        }
        let tries = [Vec2::new(step, 0.0), Vec2::new(-step, 0.0), Vec2::new(0.0, step), Vec2::new(0.0, -step)];
        let found = tries.iter().map(|&d| (best + d, ring_error(img, px, best + d))).filter(|t| t.1 < err).min_by(|a, b| a.1.total_cmp(&b.1));
        match found {
            Some((p, e)) => (best, err) = (p, e),
            None => step /= 2.0,
        }
    }
    best
}

/// The geometry a gradient is fitted along: a direction, or a centre (pixel coordinates).
#[derive(Clone, Copy)]
enum Shape {
    Linear(Vec2),
    Radial(Point),
}

/// The gradient of `shape` that the pixels `px` of `img` are, with as few stops as keep the ramps
/// within [`STOP_TOLERANCE`] of their colours; `None` when they are one flat colour.
fn fit(img: &Raster, px: &[usize], shape: Shape) -> Option<TracedGradient> {
    let w = img.width as usize;
    let at = |i: usize| Point::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
    let rgb = |i: usize| img.rgba.get(i * 4..i * 4 + 3).map_or([0.0; 3], |p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
    let along = |i: usize| match shape {
        Shape::Linear(dir) => at(i).to_vec2().dot(dir),
        Shape::Radial(c) => (at(i) - c).hypot(),
    };
    let mut ts: Vec<f64> = px.iter().map(|&i| along(i)).collect();
    ts.sort_unstable_by(f64::total_cmp);
    // The ends: the 1st and 99th percentiles (the few pixels at a pointed tip are its rim).
    let pick = |q: f64| ts.get(((ts.len().saturating_sub(1)) as f64 * q).round() as usize).copied();
    let (lo, hi) = (pick(0.01)?, pick(0.99)?);
    let length = hi - lo;
    if !(2.0..1e6).contains(&length) {
        return None;
    }
    let bins = (length.ceil() as usize).clamp(2, 4096);
    // Per bin: the colour sum, the pixels and the sum of their places (0..1 along the range).
    let mut sums = vec![([0.0; 3], 0.0f64, 0.0f64); bins];
    for &i in px {
        let u = (along(i) - lo) / length;
        let t = (u * bins as f64).floor().clamp(0.0, (bins - 1) as f64) as usize;
        if let Some(b) = sums.get_mut(t) {
            let c = rgb(i);
            (0..3).for_each(|k| b.0[k] += c[k]);
            b.1 += 1.0;
            b.2 += u.clamp(0.0, 1.0);
        }
    }
    // Each bin's mean colour at the mean place of its pixels, smoothed over its neighbours
    // (weighted by their pixels).
    let mut curve: Vec<(f64, [f64; 3])> = vec![];
    for k in 0..bins {
        if sums.get(k).is_none_or(|b| b.1 <= 0.0) {
            continue;
        }
        let (mut c, mut n, mut place) = ([0.0; 3], 0.0, 0.0);
        for b in sums.get(k.saturating_sub(2)..(k + 3).min(bins)).unwrap_or(&[]) {
            (0..3).for_each(|ch| c[ch] += b.0[ch]);
            n += b.1;
            place += b.2;
        }
        if n > 0.0 {
            curve.push((place / n, c.map(|v| v / n)));
        }
    }
    let change = |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max);
    let (lo_c, hi_c) = curve
        .iter()
        .fold(([f64::MAX; 3], [f64::MIN; 3]), |m, (_, c)| (std::array::from_fn(|k| m.0[k].min(c[k])), std::array::from_fn(|k| m.1[k].max(c[k]))));
    if curve.len() < 2 || change(lo_c, hi_c) < MIN_CHANGE {
        return None;
    }
    let mut tol = STOP_TOLERANCE;
    let keep = loop {
        let mut keep = vec![0, curve.len() - 1];
        loop {
            let mut worst = (tol, 0);
            for pair in keep.windows(2) {
                let (Some(&(ta, a)), Some(&(tb, b))) = (curve.get(pair[0]), curve.get(pair[1])) else { return None };
                for (j, &(t, c)) in curve.iter().enumerate().take(pair[1]).skip(pair[0] + 1) {
                    let k = if tb > ta { (t - ta) / (tb - ta) } else { 0.0 };
                    let off = (0..3).map(|ch| (c[ch] - (a[ch] + (b[ch] - a[ch]) * k)).abs()).fold(0.0, f64::max);
                    if off > worst.0 {
                        worst = (off, j);
                    }
                }
            }
            if worst.1 == 0 || keep.len() > MAX_STOPS {
                break;
            }
            keep.push(worst.1);
            keep.sort_unstable();
        }
        if keep.len() <= MAX_STOPS || tol > 64.0 {
            break keep;
        }
        tol *= 1.5;
    };
    let stops: Vec<(f64, [u8; 3])> =
        keep.iter().filter_map(|&j| curve.get(j)).map(|&(t, c)| (t, c.map(|v| v.round().clamp(0.0, 255.0) as u8))).collect();
    let (first, last) = (stops.first()?.0, stops.last()?.0);
    if last - first <= 0.0 {
        return None;
    }
    match shape {
        Shape::Linear(dir) => {
            let point = |t: f64| Point::ZERO + dir * (lo + t * length);
            Some(TracedGradient {
                start: point(first),
                end: point(last),
                stops: stops.iter().map(|&(t, c)| ((t - first) / (last - first), c)).collect(),
                radial: false,
            })
        }
        Shape::Radial(c) => {
            // Stops at their distances from the centre, out to the last one's.
            let radius = lo + last * length;
            if !(radius.is_finite() && radius > 0.0) {
                return None;
            }
            Some(TracedGradient {
                start: c,
                end: c + Vec2::new(radius, 0.0),
                stops: stops.iter().map(|&(t, col)| (((lo + t * length) / radius).clamp(0.0, 1.0), col)).collect(),
                radial: true,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(w: u32, h: u32, f: impl Fn(f64, f64) -> [f64; 3]) -> Moments {
        let mut m = Moments::ZERO;
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64 / 20.0 - 1.0, y as f64 / 20.0 - 0.5);
                m.add(fx, fy, f(fx, fy));
            }
        }
        m
    }

    #[test]
    fn a_parabola_along_x_fits_exactly() {
        let m = moments(40, 20, |x, _| {
            let v = 100.0 + 50.0 * x - 80.0 * x * x;
            [v, 255.0 - v, 30.0]
        });
        let (e, model) = m.best();
        assert!(e < 1e-3, "{e}");
        assert!(matches!(model, Some(Model::Linear { u }) if u.x.abs() > 0.999), "{model:?}");
    }

    #[test]
    fn rings_about_a_centre_are_radial() {
        let m = moments(40, 40, |x, y| {
            let r2 = (x - 1.5).powi(2) + (y + 0.7).powi(2);
            [40.0 + 30.0 * r2, 200.0 - 20.0 * r2, 90.0]
        });
        let (e, model) = m.best();
        assert!(e < 1e-3, "{e}");
        let Some(Model::Radial { centre }) = model else { panic!("{model:?}") };
        assert!((centre - Point::new(1.5, -0.7)).hypot() < 1e-3, "{centre:?}");
    }

    #[test]
    fn bands_of_a_gradient_merge_and_an_edge_does_not() {
        // Left: a ramp from dark to light blue cut into 6 bands; right: a flat orange.
        let (w, h) = (120u32, 40u32);
        let img = Raster::from_fn(w, h, |x, _| if x < 80 { [20, (40 + x * 2) as u8, (100 + x) as u8, 255] } else { [240, 130, 20, 255] });
        let labels: Vec<u16> = (0..w * h).map(|i| if i % w < 80 { ((i % w) / 14) as u16 } else { 6 }).collect();
        let mut palette: Vec<[u8; 3]> = (0..6).map(|b| [20, (40 + b * 28 + 14) as u8, (100 + b * 14 + 7) as u8]).collect();
        palette.push([240, 130, 20]);
        let mut q = Quantized { labels, palette };
        let mut gradients = vec![None; q.palette.len()];
        merge_gradients(&img, &mut q, &mut gradients);
        let left = q.labels[(w / 2 - 30) as usize];
        assert!((0..80).all(|x| q.labels[x as usize] == left), "the bands are one area");
        assert_eq!(q.labels[100], 6, "the orange stays its own");
        let g = gradients[left as usize].as_ref().expect("a gradient");
        let c = g.color_at(Point::new(10.5, 5.0));
        assert!((c[1] - 61.0).abs() < 4.0, "{c:?}");
    }

    #[test]
    fn curved_bands_merge_into_a_radial_gradient() {
        // A ring segment shaded by distance from (−40, 60): bands cut by the palette.
        let (w, h) = (100u32, 60u32);
        let r = |x: u32, y: u32| ((x as f64 + 0.5 + 40.0).powi(2) + (y as f64 + 0.5 - 60.0).powi(2)).sqrt();
        let img = Raster::from_fn(w, h, |x, y| {
            let t = ((r(x, y) - 40.0) / 120.0).clamp(0.0, 1.0);
            [(20.0 + 200.0 * t) as u8, (80.0 + 100.0 * t) as u8, 200, 255]
        });
        let labels: Vec<u16> = (0..w * h).map(|i| ((r(i % w, i / w) - 40.0) / 15.0).clamp(0.0, 7.0) as u16).collect();
        let palette: Vec<[u8; 3]> = (0..8).map(|b| [(20 + b * 25) as u8, (80 + b * 12) as u8, 200]).collect();
        let mut q = Quantized { labels, palette };
        let mut gradients = vec![None; q.palette.len()];
        merge_gradients(&img, &mut q, &mut gradients);
        let first = q.labels[0];
        assert!(q.labels.iter().all(|&l| l == first), "one area");
        let g = gradients[first as usize].as_ref().expect("a gradient");
        assert!(g.radial, "{g:?}");
        for (x, y) in [(5, 5), (50, 30), (95, 55)] {
            let want = img.pixel(x, y);
            let got = g.color_at(Point::new(x as f64 + 0.5, y as f64 + 0.5));
            assert!((got[0] - f64::from(want[0])).abs() < 5.0, "({x},{y}) {got:?} vs {want:?}");
        }
    }
}
