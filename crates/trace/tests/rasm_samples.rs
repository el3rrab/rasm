//! Rasm's yardstick on real images: traces every image in `TRACE_SAMPLES=dir` (files their owner
//! keeps out of the repository) with Image Trace's 16 Colors and with Gradient Logo, and reports how
//! close each result is to its image.
//!
//! ```text
//! TRACE_SAMPLES=$PWD/plan/samples [TRACE_SAMPLES_OUT=dir] \
//!   cargo test --release -p vectorcraft-trace --test rasm_samples -- --ignored --nocapture
//! ```
//!
//! Per image and setting: ΔE (OKLab × 100) over the art (the pixels that differ from the
//! background) and over all pixels, paths, anchors, gradient fills and time. `TRACE_SAMPLES_OUT`
//! also writes each result drawn as PNG.
// Integration tests: unwrapping and panicking on failure is fine here, unlike in shipped code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "quality/raster.rs"]
#[allow(dead_code)]
mod raster;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use kurbo::Point;
use vectorcraft_trace::rasm::{GradientLogo, ShapeGradient, trace_gradient_logo};
use vectorcraft_trace::{Raster, preset, trace};

/// Colour difference: Euclidean in Oklab, × 100.
fn delta(a: [u8; 3], b: [u8; 3]) -> f64 {
    let lab = |c: [u8; 3]| {
        let lin = c.map(|v| {
            let v = f64::from(v) / 255.0;
            if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
        });
        let l = (0.412_221_470_8 * lin[0] + 0.536_332_536_3 * lin[1] + 0.051_445_992_9 * lin[2]).cbrt();
        let m = (0.211_903_498_2 * lin[0] + 0.680_699_545_1 * lin[1] + 0.107_396_956_6 * lin[2]).cbrt();
        let s = (0.088_302_461_9 * lin[0] + 0.281_718_837_6 * lin[1] + 0.629_978_700_5 * lin[2]).cbrt();
        [
            0.210_454_255_3 * l + 0.793_617_785 * m - 0.004_072_046_8 * s,
            1.977_998_495_1 * l - 2.428_592_205 * m + 0.450_593_709_9 * s,
            0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766 * s,
        ]
    };
    let (a, b) = (lab(a), lab(b));
    100.0 * ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// `img` over white.
fn over_white(img: &Raster) -> Raster {
    let mut out = img.clone();
    for p in out.rgba.as_chunks_mut::<4>().0 {
        let a = u32::from(p[3]);
        for v in p.iter_mut().take(3) {
            *v = ((u32::from(*v) * a + 255 * (255 - a)) / 255) as u8;
        }
        p[3] = 255;
    }
    out
}

/// Fills (bottom first) with flat colours or gradients, painted over `bg`.
fn draw(scene: &[(kurbo::BezPath, [u8; 3], Option<&ShapeGradient>)], bg: [u8; 3], w: usize, h: usize) -> Raster {
    let mut buf = vec![bg.map(f32::from); w * h];
    let mut cov = vec![0.0f32; w * h];
    for (path, color, gradient) in scene {
        let (r0, r1) = raster::coverage_into(path, w, h, &mut cov);
        for i in r0 * w..r1 * w {
            let a = cov[i].min(1.0);
            if a > 0.0 {
                let c = match gradient {
                    Some(g) => g.color_at(Point::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5)).map(|v| v as f32),
                    None => color.map(f32::from),
                };
                for ch in 0..3 {
                    buf[i][ch] += (c[ch] - buf[i][ch]) * a;
                }
                cov[i] = 0.0;
            }
        }
    }
    Raster::from_fn(w as u32, h as u32, |x, y| {
        let c = buf[y as usize * w + x as usize];
        [c[0].round().clamp(0.0, 255.0) as u8, c[1].round().clamp(0.0, 255.0) as u8, c[2].round().clamp(0.0, 255.0) as u8, 255]
    })
}

/// (ΔE over the art, ΔE over every pixel).
fn fidelity(src: &Raster, out: &Raster, bg: [u8; 3]) -> (f64, f64) {
    let (mut art, mut n_art, mut all) = (0.0, 0usize, 0.0);
    for y in 0..src.height {
        for x in 0..src.width {
            let (a, b) = (src.pixel(x, y), out.pixel(x, y));
            let (a, b) = ([a[0], a[1], a[2]], [b[0], b[1], b[2]]);
            let d = delta(a, b);
            all += d;
            if delta(a, bg) > 3.0 {
                art += d;
                n_art += 1;
            }
        }
    }
    (art / n_art.max(1) as f64, all / (src.width * src.height).max(1) as f64)
}

fn border_color(img: &Raster) -> [u8; 3] {
    let mut sum = [0u64; 3];
    let mut n = 0u64;
    for x in 0..img.width {
        for y in [0, img.height - 1] {
            let p = img.pixel(x, y);
            (0..3).for_each(|k| sum[k] += u64::from(p[k]));
            n += 1;
        }
    }
    sum.map(|s| (s / n.max(1)) as u8)
}

fn images(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp"))
        })
        .collect();
    v.sort();
    v
}

#[test]
#[ignore = "needs TRACE_SAMPLES=dir of real images (kept out of the repository); run with --release"]
fn rasm_samples_report() {
    let Some(dir) = std::env::var_os("TRACE_SAMPLES") else { return };
    let out = std::env::var_os("TRACE_SAMPLES_OUT").map(PathBuf::from);
    let mut report =
        String::from("| Image | Setting | ΔE art | ΔE all | Paths | Anchors | Gradients | s |\n|---|---|---:|---:|---:|---:|---:|---:|\n");
    let mut totals: Vec<(&str, f64, f64, usize, usize)> = vec![("16 Colors", 0.0, 0.0, 0, 0), ("Gradient Logo", 0.0, 0.0, 0, 0)];
    let files = images(Path::new(&dir));
    for file in &files {
        let img = over_white(&Raster::decode(&std::fs::read(file).unwrap()).unwrap());
        let name = file.file_stem().unwrap().to_string_lossy().into_owned();
        let (w, h) = (img.width as usize, img.height as usize);
        let bg = border_color(&img);
        for (k, setting) in ["16 Colors", "Gradient Logo"].into_iter().enumerate() {
            let t = Instant::now();
            let scene: Vec<(kurbo::BezPath, [u8; 3], Option<ShapeGradient>)> = if k == 0 {
                trace(&img, &preset("16 Colors").unwrap()).paths.iter().map(|p| (p.path.to_bezpath(), p.color, None)).collect()
            } else {
                trace_gradient_logo(&img, &GradientLogo::default(), usize::MAX)
                    .unwrap()
                    .into_iter()
                    .map(|s| (s.path.path.to_bezpath(), s.path.color, s.gradient))
                    .collect()
            };
            let secs = t.elapsed().as_secs_f64();
            let refs: Vec<(kurbo::BezPath, [u8; 3], Option<&ShapeGradient>)> = scene.iter().map(|(p, c, g)| (p.clone(), *c, g.as_ref())).collect();
            let drawn = draw(&refs, bg, w, h);
            let (art, all) = fidelity(&img, &drawn, bg);
            if let Some(dir) = &out {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(dir.join(format!("{name} — {setting}.png")), drawn.encode_png()).unwrap();
            }
            let anchors: usize = scene.iter().map(|s| s.0.elements().len()).sum();
            let gradients = scene.iter().filter(|s| s.2.is_some()).count();
            let _ = writeln!(report, "| {name} | {setting} | {art:.2} | {all:.2} | {} | {anchors} | {gradients} | {secs:.2} |", scene.len());
            let t = &mut totals[k];
            (t.1, t.2, t.3, t.4) = (t.1 + art, t.2 + all, t.3 + scene.len(), t.4 + gradients);
        }
    }
    let n = files.len().max(1) as f64;
    for (setting, art, all, paths, gradients) in totals {
        let _ = writeln!(
            report,
            "| **average** | {setting} | {:.2} | {:.2} | {:.0} | | {:.0} | |",
            art / n,
            all / n,
            paths as f64 / n,
            gradients as f64 / n
        );
    }
    println!("{report}");
}
