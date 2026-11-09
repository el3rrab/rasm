//! Rasm, our own tracing (kept apart from the tracer's own files, which it only uses):
//!
//! - [`trace_gradient_logo`]: Gradient Logo. The image is cut into its shapes by its edges
//!   ([`segment`]), so a shaded shape is one path however its colour changes inside it, and each
//!   is filled with the linear or radial gradient (or flat colour) of its own pixels.
//! - [`paint_pixels`] and [`pixels_inside`]: the gradient an area of an image has, which Gradient
//!   from Image fills selected shapes with.

mod fit;
mod segment;

use std::sync::atomic::AtomicUsize;

pub use fit::{GradientChoice, PixelPaint, ShapeGradient, Stops, paint_pixels, pixels_inside};

use crate::layers::{Layers, parallel_map_at_most};
use crate::pipeline::bw_tones;
use crate::{Method, Mode, Raster, TraceError, TraceParams, TracedPath};

/// Gradient Logo's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientLogo {
    /// The smallest shape kept (px); smaller ones join the neighbour they have the weakest edge with.
    pub noise: u32,
    /// How closely the curves follow the edges (0–100), and the turn that makes a corner (0–100),
    /// as Image Trace's Paths and Corners.
    pub paths: f64,
    pub corners: f64,
}

impl Default for GradientLogo {
    fn default() -> Self {
        GradientLogo { noise: 30, paths: 70.0, corners: 60.0 }
    }
}

/// A shape Gradient Logo traced: its outline (filled with `path.color` where it has no gradient)
/// and its gradient.
#[derive(Clone, Debug, PartialEq)]
pub struct GradientShape {
    pub path: TracedPath,
    pub gradient: Option<ShapeGradient>,
}

/// The shapes of `img` (bottom-most first), as described in the module documentation; at most
/// `max_anchors` anchor points ([`TraceError::TooComplex`] past them).
pub fn trace_gradient_logo(img: &Raster, settings: &GradientLogo, max_anchors: usize) -> Result<Vec<GradientShape>, TraceError> {
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || w.checked_mul(h).and_then(|n| n.checked_mul(4)) != Some(img.rgba.len()) {
        return Ok(vec![]);
    }
    let Some((q, gradients)) = segment::segment(img, settings.noise as usize) else {
        return Err(TraceError::TooComplex { max: max_anchors });
    };
    let params = TraceParams {
        mode: Mode::Color,
        paths: settings.paths,
        corners: settings.corners,
        noise: settings.noise,
        method: Method::Abutting,
        ..TraceParams::default()
    };
    // Pixel count per shape, and the box its pixels lie in.
    let mut counts = vec![0usize; q.palette.len()];
    let mut boxes: Vec<Option<(usize, usize, usize, usize)>> = vec![None; q.palette.len()];
    for (i, &l) in q.labels.iter().enumerate() {
        if let (Some(c), Some(b)) = (counts.get_mut(l as usize), boxes.get_mut(l as usize)) {
            *c += 1;
            let (x, y) = (i % w, i / w);
            *b = Some(b.map_or((x, y, x + 1, y + 1), |b| (b.0.min(x), b.1.min(y), b.2.max(x + 1), b.3.max(y + 1))));
        }
    }
    // Bottom → top: the largest first.
    let mut order: Vec<usize> = (0..q.palette.len()).filter(|&i| counts.get(i).is_some_and(|&c| c > 0)).collect();
    order.sort_by(|a, b| counts.get(*b).cmp(&counts.get(*a)).then(a.cmp(b)));
    let mut rank = vec![usize::MAX; q.palette.len()];
    for (r, &i) in order.iter().enumerate() {
        if let Some(slot) = rank.get_mut(i) {
            *slot = r;
        }
    }
    // Edges are placed between the shapes' own colours (their gradients' middles).
    let flat = vec![None; q.palette.len()];
    let job = Layers {
        q: &q,
        rank: &rank,
        params: &params,
        img,
        ignore: None,
        boxes,
        gradients: &flat,
        bw_tones: bw_tones(img, &q),
        style: params.outline_style(),
        w,
        h,
        anchors: AtomicUsize::new(0),
        max_anchors,
    };
    let largest = job.boxes.iter().flatten().map(|b| (b.2 - b.0) * (b.3 - b.1)).max().unwrap_or(1).max(1);
    let at_a_time = ((512 << 20) / (largest * 10)).max(1);
    let layers: Vec<(usize, usize)> = order.iter().copied().enumerate().collect();
    let traced = parallel_map_at_most(at_a_time, &layers, |&(r, ci)| job.layer(r, ci));
    let mut out = vec![];
    for (paths, &(_, ci)) in traced.into_iter().zip(&layers) {
        let paths = paths.ok_or(TraceError::TooComplex { max: max_anchors })?;
        let gradient = gradients.get(ci).cloned().flatten();
        out.extend(paths.into_iter().map(|path| GradientShape { path, gradient: gradient.clone() }));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_shaded_shapes_trace_as_two_gradient_shapes() {
        let img = Raster::from_fn(120, 60, |x, _| {
            if x < 60 { [(150 + x) as u8, (60 + x * 2) as u8, 20, 255] } else { [(250 - (x - 60)) as u8, (40 + x - 60) as u8, 90, 255] }
        });
        let shapes = trace_gradient_logo(&img, &GradientLogo::default(), usize::MAX).unwrap();
        assert_eq!(shapes.len(), 2, "{shapes:?}");
        assert!(shapes.iter().all(|s| s.gradient.is_some()));
    }
}
