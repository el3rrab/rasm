//! Real-image yardstick: traces every image in `TRACE_SAMPLES=dir` (files the user owns, kept
//! out of the repository) with a few settings and reports how close each result is to its image.
//!
//! ```text
//! TRACE_SAMPLES=plan/samples [TRACE_SAMPLES_OUT=dir] \
//!   cargo test --release -p vectorcraft-trace --test quality samples_report -- --ignored --nocapture
//! ```
//!
//! Per image and setting: mean colour difference (OKLab ΔE×100, all pixels and edge-weighted),
//! paths, anchors, gradient fills and time. `TRACE_SAMPLES_OUT` also writes each result drawn as
//! PNG, for looking at.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use kurbo::BezPath;
use vectorcraft_trace::{Raster, TraceParams, TraceResult, TracedGradient, preset, trace};

use crate::measure::{delta, fidelity, painted};
use crate::raster::render_painted;

/// The settings compared: a name and the parameters.
pub fn settings() -> Vec<(&'static str, TraceParams)> {
    let with_gradients = |name: &str| TraceParams { gradients: true, ..preset(name).unwrap() };
    vec![
        ("16 Colors", preset("16 Colors").unwrap()),
        ("16 Colors + Gradients", with_gradients("16 Colors")),
        ("Auto Color + Gradients", with_gradients("Auto Color")),
        ("High Fidelity Photo + Gradients", with_gradients("High Fidelity Photo")),
        ("Gradient Logo", preset("Gradient Logo").unwrap()),
    ]
}

/// The images of `dir`, by name.
pub fn images(dir: &Path) -> Vec<PathBuf> {
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

/// `res` drawn over white (where nothing is traced, the image's own background is white or
/// transparent in practice; the colour difference counts it either way).
pub fn draw(res: &TraceResult, w: usize, h: usize, bg: [u8; 3]) -> Raster {
    let scene: Vec<(BezPath, [u8; 3], Option<&TracedGradient>)> = res.paths.iter().map(|p| (painted(p), p.color, p.gradient.as_ref())).collect();
    render_painted(&scene, bg, w, h)
}

/// `img` composited over white: a transparent background is shown white, as an editor shows it.
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

/// The mean colour difference over the pixels of `src` that are not background `bg`.
fn ink_delta(src: &Raster, out: &Raster, bg: [u8; 3]) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for y in 0..src.height {
        for x in 0..src.width {
            let (a, b) = (src.pixel(x, y), out.pixel(x, y));
            let a = [a[0], a[1], a[2]];
            if delta(a, bg) > 3.0 {
                sum += delta(a, [b[0], b[1], b[2]]);
                n += 1;
            }
        }
    }
    if n > 0 { sum / n as f64 } else { 0.0 }
}

/// The colour most of the image's border has: what an untraced area shows through to.
fn border_color(img: &Raster) -> [u8; 3] {
    let (w, h) = (img.width, img.height);
    let mut sum = [0u64; 3];
    let mut n = 0u64;
    for x in 0..w {
        for y in [0, h - 1] {
            let p = img.pixel(x, y);
            (0..3).for_each(|k| sum[k] += u64::from(p[k]));
            n += 1;
        }
    }
    sum.map(|s| (s / n.max(1)) as u8)
}

/// One image traced with one setting.
pub struct Row {
    pub image: String,
    pub setting: &'static str,
    pub mean: f64,
    pub edge: f64,
    /// Mean ΔE over the art only: the pixels that differ from the background.
    pub ink: f64,
    pub paths: usize,
    pub anchors: usize,
    pub gradients: usize,
    pub secs: f64,
}

/// Traces `path` with every setting; writes the drawings to `out` when given.
pub fn measure_image(path: &Path, out: Option<&Path>) -> Vec<Row> {
    let bytes = std::fs::read(path).unwrap();
    let img = over_white(&Raster::decode(&bytes).unwrap());
    let name = path.file_stem().unwrap().to_string_lossy().into_owned();
    let (w, h) = (img.width as usize, img.height as usize);
    let bg = border_color(&img);
    settings()
        .into_iter()
        .map(|(setting, params)| {
            let t = Instant::now();
            let res = trace(&img, &params);
            let secs = t.elapsed().as_secs_f64();
            let drawn = draw(&res, w, h, bg);
            let (mean, edge) = fidelity(&img, &drawn);
            let ink = ink_delta(&img, &drawn, bg);
            if let Some(dir) = out {
                std::fs::create_dir_all(dir).unwrap();
                let file = dir.join(format!("{name} — {setting}.png"));
                std::fs::write(file, drawn.encode_png()).unwrap();
            }
            Row {
                image: name.clone(),
                setting,
                mean,
                edge,
                ink,
                paths: res.paths.len(),
                anchors: res.anchor_count(),
                gradients: res.paths.iter().filter(|p| p.gradient.is_some()).count(),
                secs,
            }
        })
        .collect()
}

/// The report as a Markdown table, with each setting's average last.
pub fn report(rows: &[Row]) -> String {
    let mut s = String::from(
        "| Image | Setting | ΔE art | ΔE mean | ΔE edges | Paths | Anchors | Gradients | s |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for r in rows {
        let _ = writeln!(
            s,
            "| {} | {} | {:.2} | {:.2} | {:.2} | {} | {} | {} | {:.2} |",
            r.image, r.setting, r.ink, r.mean, r.edge, r.paths, r.anchors, r.gradients, r.secs
        );
    }
    for (setting, _) in settings() {
        let of: Vec<&Row> = rows.iter().filter(|r| r.setting == setting).collect();
        let n = of.len().max(1) as f64;
        let avg = |f: fn(&Row) -> f64| of.iter().map(|r| f(r)).sum::<f64>() / n;
        let _ = writeln!(
            s,
            "| **average** | {setting} | {:.2} | {:.2} | {:.2} | {:.0} | {:.0} | {:.1} | {:.2} |",
            avg(|r| r.ink),
            avg(|r| r.mean),
            avg(|r| r.edge),
            avg(|r| r.paths as f64),
            avg(|r| r.anchors as f64),
            avg(|r| r.gradients as f64),
            avg(|r| r.secs)
        );
    }
    s
}
