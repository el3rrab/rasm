//! Shapes by edges: the image is cut into areas where it has an edge (a sharp change of colour
//! over a pixel or two), not where its colours cross from one palette entry to the next, so a
//! shaded shape (a feather light along one side and dark along the other) is one area however much
//! its colour changes inside it. Each area is then painted with the gradient or flat colour its own
//! pixels have ([`crate::paint_pixels`]).
//!
//! 1. The image is blurred a little (noise is no edge) and each pixel gets its edge strength: the
//!    largest change of a channel across it (Sobel), in levels per pixel.
//! 2. Seeds: the connected runs of pixels with almost no edge strength.
//! 3. Watershed: the seeds grow over the rest, weakest edges first, until every pixel belongs to
//!    one; two seeds meet on the strongest edge between them.
//! 4. Areas whose shared border is no edge (the seeds of one shape that a little texture parted)
//!    are merged, weakest border first, and areas smaller than the noise size join the neighbour
//!    they have the weakest border with.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use kurbo::Point;

use crate::quantize::{Quantized, TRANSPARENT};
use crate::{GradientChoice, Raster, Stops, TracedGradient, paint_pixels};

/// Edge strength (levels a pixel) below which a pixel can seed an area, and the fewest pixels a
/// seed has.
const SEED_EDGE: f32 = 2.0;
const SEED_PIXELS: usize = 12;
/// The mean edge strength along a border below which the two areas are one.
const WEAK_BORDER: f64 = 6.0;
/// The most areas (palette labels are 16 bits).
const MAX_AREAS: usize = 60_000;

/// The pixels beside pixel `i` of a `w` × `h` image.
fn beside(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % w, i / w);
    [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten()
}

/// A 3 × 3 box blur of the RGB channels, twice (about a Gaussian of radius 1), edges clamped.
fn blurred(img: &Raster) -> Vec<[f32; 3]> {
    let (w, h) = (img.width as usize, img.height as usize);
    let mut a: Vec<[f32; 3]> = img.rgba.as_chunks::<4>().0.iter().map(|p| [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])]).collect();
    let mut b = a.clone();
    for _ in 0..2 {
        for y in 0..h {
            for x in 0..w {
                let mut s = [0.0f32; 3];
                for dy in [-1i64, 0, 1] {
                    for dx in [-1i64, 0, 1] {
                        let (nx, ny) = ((x as i64 + dx).clamp(0, w as i64 - 1) as usize, (y as i64 + dy).clamp(0, h as i64 - 1) as usize);
                        if let Some(c) = a.get(ny * w + nx) {
                            (0..3).for_each(|k| s[k] += c[k]);
                        }
                    }
                }
                if let Some(o) = b.get_mut(y * w + x) {
                    *o = s.map(|v| v / 9.0);
                }
            }
        }
        std::mem::swap(&mut a, &mut b);
    }
    a
}

/// Each pixel's edge strength: the largest Sobel gradient magnitude of a channel, in levels a pixel.
fn edge_strength(c: &[[f32; 3]], w: usize, h: usize) -> Vec<f32> {
    let at = |x: i64, y: i64| c.get(y.clamp(0, h as i64 - 1) as usize * w + x.clamp(0, w as i64 - 1) as usize).copied().unwrap_or([0.0; 3]);
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            let (tl, t, tr, l, r, bl, b, br) =
                (at(x - 1, y - 1), at(x, y - 1), at(x + 1, y - 1), at(x - 1, y), at(x + 1, y), at(x - 1, y + 1), at(x, y + 1), at(x + 1, y + 1));
            (0..3)
                .map(|k| {
                    let gx = (tr[k] + 2.0 * r[k] + br[k]) - (tl[k] + 2.0 * l[k] + bl[k]);
                    let gy = (bl[k] + 2.0 * b[k] + br[k]) - (tl[k] + 2.0 * t[k] + tr[k]);
                    gx.hypot(gy) / 8.0
                })
                .fold(0.0, f32::max)
        })
        .collect()
}

/// The area of each pixel (`u32::MAX`: transparent), and how many areas there are.
fn areas(img: &Raster, min_pixels: usize) -> (Vec<u32>, usize) {
    let (w, h) = (img.width as usize, img.height as usize);
    let opaque = |i: usize| img.rgba.get(i * 4 + 3).is_some_and(|&a| a >= 128);
    let g = edge_strength(&blurred(img), w, h);
    let mut of = vec![u32::MAX; w * h];
    // Seeds.
    let mut count = 0u32;
    let (mut stack, mut members) = (vec![], vec![]);
    let mut seen = vec![false; w * h];
    for seed in 0..w * h {
        if seen[seed] || !opaque(seed) || g.get(seed).is_none_or(|&e| e >= SEED_EDGE) {
            continue;
        }
        members.clear();
        seen[seed] = true;
        stack.push(seed);
        while let Some(i) = stack.pop() {
            members.push(i);
            for j in beside(i, w, h) {
                if !seen[j] && opaque(j) && g.get(j).is_some_and(|&e| e < SEED_EDGE) {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if members.len() >= SEED_PIXELS {
            members.iter().for_each(|&i| of[i] = count);
            count += 1;
        }
    }
    if count == 0 {
        // No flat place at all: one area of every opaque pixel.
        (0..w * h).filter(|&i| opaque(i)).for_each(|i| of[i] = 0);
        return (of, usize::from((0..w * h).any(opaque)));
    }
    // Watershed by a bucket queue on the edge strength (quarter levels).
    let bucket = |i: usize| (g.get(i).copied().unwrap_or(0.0) * 4.0).clamp(0.0, 1023.0) as usize;
    let mut queue: Vec<Vec<usize>> = vec![vec![]; 1024];
    for i in 0..w * h {
        if of[i] != u32::MAX && beside(i, w, h).any(|j| of[j] == u32::MAX && opaque(j)) {
            queue[0].push(i);
        }
    }
    let mut level = 0;
    loop {
        while level < queue.len() && queue[level].is_empty() {
            level += 1;
        }
        let Some(i) = queue.get_mut(level).and_then(Vec::pop) else { break };
        let label = of[i];
        for j in beside(i, w, h) {
            if of[j] == u32::MAX && opaque(j) {
                of[j] = label;
                queue[bucket(j).max(level)].push(j);
            }
        }
    }
    merge_weak(&mut of, &g, w, h, count as usize, min_pixels)
}

/// A candidate merge, weakest first: (border strength, a, b, a's version, b's version).
type Candidate = Reverse<(u64, u32, u32, u32, u32)>;

/// Merges areas across weak borders and areas under `min_pixels` into a neighbour (see the module
/// documentation); the areas numbered afresh from 0.
fn merge_weak(of: &mut [u32], g: &[f32], w: usize, h: usize, count: usize, min_pixels: usize) -> (Vec<u32>, usize) {
    let mut size = vec![0usize; count];
    of.iter().filter(|&&a| a != u32::MAX).for_each(|&a| size[a as usize] += 1);
    // Borders: (pairs, summed strength).
    let mut borders: Vec<HashMap<u32, (f64, f64)>> = vec![HashMap::new(); count];
    for i in 0..w * h {
        let a = of[i];
        if a == u32::MAX {
            continue;
        }
        let right = (i % w + 1 < w).then(|| i + 1);
        let down = (i / w + 1 < h).then(|| i + w);
        for j in [right, down].into_iter().flatten() {
            let b = of[j];
            if b == u32::MAX || b == a {
                continue;
            }
            let s = f64::from(g[i].max(g[j]));
            for (x, y) in [(a, b), (b, a)] {
                let e = borders[x as usize].entry(y).or_insert((0.0, 0.0));
                e.0 += 1.0;
                e.1 += s;
            }
        }
    }
    let mut parent: Vec<u32> = (0..count as u32).collect();
    fn find(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            let gp = parent[parent[i as usize] as usize];
            parent[i as usize] = gp;
            i = gp;
        }
        i
    }
    let mut version = vec![0u32; count];
    // (mean strength, a, b, versions); a small area's borders all count as weak.
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
    let key = |size: &[usize], a: u32, b: u32, e: (f64, f64)| -> Option<u64> {
        let mean = e.1 / e.0.max(1.0);
        let small = size[a as usize] < min_pixels || size[b as usize] < min_pixels;
        (mean < WEAK_BORDER || small).then(|| (if small { mean / 1000.0 } else { mean }).to_bits())
    };
    for (a, map) in borders.iter().enumerate() {
        for (&b, &e) in map {
            if (a as u32) < b
                && let Some(k) = key(&size, a as u32, b, e)
            {
                heap.push(Reverse((k, a as u32, b, 0, 0)));
            }
        }
    }
    while let Some(Reverse((_, a, b, va, vb))) = heap.pop() {
        if find(&mut parent, a) != a || find(&mut parent, b) != b || version[a as usize] != va || version[b as usize] != vb {
            continue;
        }
        // The smaller into the larger.
        let (keep, gone) = if size[a as usize] >= size[b as usize] { (a, b) } else { (b, a) };
        parent[gone as usize] = keep;
        size[keep as usize] += size[gone as usize];
        version[keep as usize] += 1;
        let from = std::mem::take(&mut borders[gone as usize]);
        for (c, e) in from {
            if c == keep {
                continue;
            }
            borders[c as usize].remove(&gone);
            let x = borders[c as usize].entry(keep).or_insert((0.0, 0.0));
            x.0 += e.0;
            x.1 += e.1;
            let y = borders[keep as usize].entry(c).or_insert((0.0, 0.0));
            y.0 += e.0;
            y.1 += e.1;
        }
        borders[keep as usize].remove(&gone);
        let vk = version[keep as usize];
        let next: Vec<(u32, (f64, f64))> = borders[keep as usize].iter().map(|(&c, &e)| (c, e)).collect();
        for (c, e) in next {
            if let Some(k) = key(&size, keep, c, e) {
                let (x, y) = (keep.min(c), keep.max(c));
                let (vx, vy) = if x == keep { (vk, version[c as usize]) } else { (version[c as usize], vk) };
                heap.push(Reverse((k, x, y, vx, vy)));
            }
        }
    }
    // Numbered afresh.
    let mut number = vec![u32::MAX; count];
    let mut n = 0u32;
    for a in of.iter_mut() {
        if *a == u32::MAX {
            continue;
        }
        let r = find(&mut parent, *a) as usize;
        if number[r] == u32::MAX {
            number[r] = n;
            n += 1;
        }
        *a = number[r];
    }
    (of.to_vec(), n as usize)
}

/// How far (levels, the channel that differs most) a pixel may be from its area's paint before it
/// is taken for a detail of its own (a thin letter, a line, a dot) that no seed reached, and the
/// fewest pixels such a detail has.
/// A detail also has a core: pixels whose four neighbours are all of it (the soft rim along an
/// area's edge, a pixel or two wide, has none).
const DETAIL_OFF: f64 = 48.0;
const DETAIL_PIXELS: usize = 12;

/// The paint of each area of `of` (`count` areas): its colour and gradient, if any.
fn paints(img: &Raster, of: &[u32], count: usize) -> Vec<([u8; 3], Option<TracedGradient>)> {
    let mut pixels: Vec<Vec<usize>> = vec![vec![]; count];
    for (i, &a) in of.iter().enumerate() {
        if let Some(p) = pixels.get_mut(a as usize) {
            p.push(i);
        }
    }
    pixels
        .iter()
        .map(|px| {
            let paint = paint_pixels(img, px, GradientChoice::Auto, &Stops::DEFAULT);
            (paint.as_ref().map_or([0, 0, 0], |p| p.color), paint.and_then(|p| p.gradient))
        })
        .collect()
}

/// Splits off each detail the areas swallowed (see [`DETAIL_OFF`]) as an area of its own; the
/// areas it changed, to paint again.
fn split_details(img: &Raster, of: &mut [u32], count: &mut usize, painted: &[([u8; 3], Option<TracedGradient>)]) -> Vec<u32> {
    let (w, h) = (img.width as usize, img.height as usize);
    let off = |i: usize| -> bool {
        let Some(&a) = of.get(i) else { return false };
        let Some((color, g)) = painted.get(a as usize) else { return false };
        let p = Point::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
        let want = g.as_ref().map_or(color.map(f64::from), |g| g.color_at(p));
        let Some(c) = img.rgba.get(i * 4..i * 4 + 3) else { return false };
        (0..3).map(|k| (f64::from(c[k]) - want[k]).abs()).fold(0.0, f64::max) > DETAIL_OFF
    };
    let bad: Vec<bool> = (0..w * h).map(off).collect();
    let mut seen = vec![false; w * h];
    let (mut stack, mut members, mut changed) = (vec![], vec![], vec![]);
    for seed in 0..w * h {
        if seen[seed] || !bad[seed] {
            continue;
        }
        let area = of[seed];
        members.clear();
        seen[seed] = true;
        stack.push(seed);
        while let Some(i) = stack.pop() {
            members.push(i);
            for j in beside(i, w, h) {
                if !seen[j] && bad[j] && of[j] == area {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        let core = members.iter().any(|&i| beside(i, w, h).count() == 4 && beside(i, w, h).all(|j| bad[j] && of[j] == area));
        if core && members.len() >= DETAIL_PIXELS && *count < MAX_AREAS {
            let new = *count as u32;
            *count += 1;
            members.iter().for_each(|&i| of[i] = new);
            changed.extend([area, new]);
        }
    }
    changed.sort_unstable();
    changed.dedup();
    changed
}

/// The image cut into areas by its edges, each a palette entry painted with its own gradient or
/// flat colour; `noise` (px) is the smallest area kept. `None` when there are more areas than
/// labels (the image is a photo, not shapes).
pub(crate) fn segment(img: &Raster, noise: usize) -> Option<(Quantized, Vec<Option<TracedGradient>>)> {
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || img.rgba.len() != w * h * 4 {
        return None;
    }
    let (mut of, mut count) = areas(img, noise.max(1));
    if count > MAX_AREAS {
        return None;
    }
    let mut painted = paints(img, &of, count);
    // Details no seed reached, twice (a detail inside a detail), each area changed painted again.
    for _ in 0..2 {
        let changed = split_details(img, &mut of, &mut count, &painted);
        if changed.is_empty() {
            break;
        }
        let again = paints(img, &of, count);
        painted.resize(count, ([0, 0, 0], None));
        for a in changed {
            if let (Some(slot), Some(p)) = (painted.get_mut(a as usize), again.get(a as usize)) {
                *slot = p.clone();
            }
        }
    }
    let (palette, gradients): (Vec<[u8; 3]>, Vec<Option<TracedGradient>>) = painted.into_iter().unzip();
    let labels = of.iter().map(|&a| if a == u32::MAX { TRANSPARENT } else { u16::try_from(a).unwrap_or(TRANSPARENT) }).collect();
    Some((Quantized { labels, palette }, gradients))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two shaded shapes side by side, each a ramp that crosses the other's colours, parted by a
    /// sharp edge: two areas, each one whole.
    #[test]
    fn shaded_shapes_are_one_area_each() {
        let img = Raster::from_fn(120, 60, |x, _| {
            if x < 60 {
                // Dark to light orange.
                [(150 + x) as u8, (60 + x * 2) as u8, 20, 255]
            } else {
                // Light to dark orange: begins where the left one ends, past an edge.
                let t = x - 60;
                [(250 - t) as u8, (40 + t) as u8, 90, 255]
            }
        });
        let (q, gradients) = segment(&img, 25).unwrap();
        let (left, right) = (q.labels[30], q.labels[90]);
        assert_ne!(left, right);
        for y in 0..60 {
            for x in 0..120 {
                let want = if x < 59 {
                    left
                } else if x > 61 {
                    right
                } else {
                    continue;
                };
                assert_eq!(q.labels[y * 120 + x], want, "({x},{y})");
            }
        }
        assert!(gradients[left as usize].is_some() && gradients[right as usize].is_some());
    }

    /// Thin dark strokes on white (letters too thin to seed an area) are areas of their own.
    #[test]
    fn thin_strokes_are_kept() {
        let img = Raster::from_fn(80, 40, |x, y| {
            if (10..13).contains(&x) && (5..35).contains(&y) || (30..60).contains(&x) && (18..21).contains(&y) { [20, 20, 40, 255] } else { [255; 4] }
        });
        let (q, _) = segment(&img, 25).unwrap();
        let (bar, line, white) = (q.labels[20 * 80 + 11], q.labels[19 * 80 + 45], q.labels[2 * 80 + 70]);
        assert_ne!(bar, white);
        assert_ne!(line, white);
        assert!(q.palette[bar as usize].iter().all(|&c| c < 80), "{:?}", q.palette[bar as usize]);
    }
}
