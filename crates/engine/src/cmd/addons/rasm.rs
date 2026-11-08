//! Rasm (`addon.rasm.*`), our own add-on. Gradient from Image fills the selected shapes with the
//! gradient (linear or radial) or flat colour that the image under them has there: the colours are
//! read from the image's own pixels inside the shapes, as Image Trace's Gradients option reads
//! them ([`vectorcraft_trace::paint_pixels`]). With Merge, the shapes become one shape first (as
//! Pathfinder › Unite makes them) with one gradient over all of it; without, each selected object
//! gets its own. It is how traced pieces of one shaded area are put back together.

use serde_json::{Value, json};
use vectorcraft_color::{Color, Gradient, GradientGeom, GradientKind, GradientPaint, GradientStop, Paint};
use vectorcraft_doc::{Document, ImageObject, Node, NodeId, NodeKind};
use vectorcraft_geom::{Affine, FillRule, PathData, Point, Rect};
use vectorcraft_pathops as po;
use vectorcraft_trace as tr;

use super::super::pathops::{node_path, pf_leaves, shape_node, style_of};
use super::super::{CommandSpec, bad, bool_or, checked_ids_or_selection, cmd, has_selection, number_or, str_param};
use crate::{EngineError, Result, Session};

/// The command.
pub const GRADIENT_FROM_IMAGE: &str = "addon.rasm.gradientFromImage";

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "addon.rasm.gradientFromImage",
        "Gradient from Image",
        ["Object", "Plug-ins", "Rasm"],
        None,
        "{ids?: [id…] (default: the selection; images among them are left out), image?: id (the image to read the colours from: an image, or an Image Trace object's image; default: the image under the shapes that most overlaps them), kind?: \"auto\"|\"linear\"|\"radial\" (auto: whichever is nearer the image, or a flat colour where the area is one), merge?: true (unite the shapes into one, as Pathfinder › Unite, painted with one gradient; false: each selected object its own), stops?: 2..16 (8, the most stops), smoothness?: 0.5..32 (3: how far, in levels of a channel, the ramps between stops may be from the image's colours; more is fewer stops)} fill the shapes with the gradient the image has under them → {ids, image, results: [{id, kind: \"linear\"|\"radial\"|\"flat\", stops, off: root-mean-square difference from the image's pixels, levels}]}",
        has_selection,
        gradient_from_image
    )]
}

/// The bounds in the document of image `i`.
fn image_bounds(i: &ImageObject) -> Rect {
    let corners =
        [(0.0, 0.0), (i.width as f64, 0.0), (0.0, i.height as f64), (i.width as f64, i.height as f64)].map(|(x, y)| i.xf * Point::new(x, y));
    corners.iter().skip(1).fold(Rect::from_points(corners[0], corners[0]), |r, p| r.union_pt(*p))
}

/// The image of node `n`: an image, or an Image Trace object's (its first child).
fn image_of(n: &Node) -> Option<&ImageObject> {
    match &n.kind {
        NodeKind::Image(i) => Some(i),
        NodeKind::Group { children, .. } => match &children.first()?.kind {
            NodeKind::Image(i) => Some(i),
            _ => None,
        },
        _ => None,
    }
}

/// The image the colours are read from: `asked`, else the one (anywhere in the document, hidden
/// ones too) whose bounds overlap `over` most.
fn find_image(d: &Document, asked: Option<NodeId>, over: Rect) -> Result<(NodeId, ImageObject)> {
    const C: &str = GRADIENT_FROM_IMAGE;
    if let Some(id) = asked {
        let n = d.node(id).ok_or(EngineError::NoNode(id))?;
        return image_of(n).map(|i| (id, i.clone())).ok_or_else(|| bad(C, "image: not an image or an Image Trace object"));
    }
    let mut best: Option<(f64, NodeId, ImageObject)> = None;
    d.walk(|n| {
        if let NodeKind::Image(i) = &n.kind {
            let overlap = image_bounds(i).intersect(over);
            let area = overlap.width().max(0.0) * overlap.height().max(0.0);
            if area > 0.0 && best.as_ref().is_none_or(|b| area > b.0) {
                best = Some((area, n.id, (**i).clone()));
            }
        }
    });
    best.map(|b| (b.1, b.2))
        .ok_or_else(|| bad(C, "no image lies under the shapes: place the image the colours come from under them (or name it with `image`)"))
}

/// A traced gradient (in the raster's pixels) as a fill, placed in the document by `xf`.
fn fill_of(p: &tr::PixelPaint, xf: Affine) -> Paint {
    let Some(g) = &p.gradient else { return Paint::solid(Color::rgb8(p.color[0], p.color[1], p.color[2])) };
    let stops = g.stops.iter().map(|&(at, c)| GradientStop::new(at as f32, Color::rgb8(c[0], c[1], c[2]))).collect();
    let kind = if g.radial { GradientKind::Radial } else { GradientKind::Linear };
    let mut paint = GradientPaint::new(Gradient::new(kind, stops));
    paint.geom = Some(GradientGeom { start: xf * g.start, end: xf * g.end, aspect: 1.0, focal: None });
    Paint::Gradient(Box::new(paint))
}

/// The shapes of object `n`, in the document, with their fill rules.
fn shapes_of(d: &Document, n: &Node) -> Vec<(PathData, FillRule)> {
    let mut leaves = vec![];
    pf_leaves(d, n, &mut leaves);
    leaves.iter().filter_map(node_path).filter(|(p, _)| !p.is_empty()).collect()
}

fn gradient_from_image(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = GRADIENT_FROM_IMAGE;
    let choice = match str_param(p, "kind").unwrap_or("auto") {
        "auto" => tr::GradientChoice::Auto,
        "linear" => tr::GradientChoice::Linear,
        "radial" => tr::GradientChoice::Radial,
        other => return Err(bad(C, format!("kind: {other:?} is not one of auto, linear, radial"))),
    };
    let merge = bool_or(p, "merge", true);
    let most = number_or(p, "stops", 8.0, C)?;
    let smoothness = number_or(p, "smoothness", 3.0, C)?;
    if !(2.0..=16.0).contains(&most) || !(0.5..=32.0).contains(&smoothness) {
        return Err(bad(C, "stops: 2..16, smoothness: 0.5..32"));
    }
    let rule = tr::Stops { tolerance: smoothness, most: most.round() as usize };
    let asked = match p.get("image") {
        Some(v) => Some(NodeId(v.as_u64().ok_or_else(|| bad(C, "image: an object id"))?)),
        None => None,
    };
    let roots = checked_ids_or_selection(s, p, C)?;
    let d = &s.doc()?.doc;
    // The selected objects with shapes (images left out), each with its shapes.
    let objects: Vec<(NodeId, Vec<(PathData, FillRule)>)> = roots
        .iter()
        .filter_map(|&id| {
            let n = d.node(id)?;
            if image_of(n).is_some() && !matches!(n.kind, NodeKind::Group { .. }) {
                return None;
            }
            let shapes = shapes_of(d, n);
            (!shapes.is_empty()).then_some((id, shapes))
        })
        .collect();
    let all: Vec<&(PathData, FillRule)> = objects.iter().flat_map(|o| &o.1).collect();
    let over = all.iter().filter_map(|(p, _)| p.bounds()).reduce(|a, b| a.union(b)).ok_or_else(|| bad(C, "select shapes to fill"))?;
    let (image_id, image) = find_image(d, asked, over)?;
    let bytes = d.images.get(&image.key).ok_or_else(|| EngineError::Other(format!("image data `{}` is missing", image.key)))?.bytes.clone();
    let raster = tr::Raster::decode_to_fit(&bytes).map_err(|e| EngineError::Other(e.to_string()))?;
    // The raster's pixels → the image's pixels → the document.
    let xf = image.xf
        * Affine::scale_non_uniform(image.width.max(1) as f64 / raster.width.max(1) as f64, image.height.max(1) as f64 / raster.height.max(1) as f64);
    if xf.determinant().abs() < 1e-12 || !xf.determinant().is_finite() {
        return Err(bad(C, "the image has no area"));
    }
    let to_pixels = xf.inverse();
    let paint_of = |shapes: &[&(PathData, FillRule)]| -> Result<tr::PixelPaint> {
        let mut px: Vec<usize> =
            shapes.iter().flat_map(|(path, _)| tr::pixels_inside(&path.transformed(to_pixels), raster.width, raster.height)).collect();
        px.sort_unstable();
        px.dedup();
        tr::paint_pixels(&raster, &px, choice, &rule).ok_or_else(|| bad(C, "the shapes cover none of the image's pixels"))
    };
    let report = |id: NodeId, p: &tr::PixelPaint| {
        let kind = match &p.gradient {
            Some(g) if g.radial => "radial",
            Some(_) => "linear",
            None => "flat",
        };
        json!({"id": id.0, "kind": kind, "stops": p.gradient.as_ref().map_or(1, |g| g.stops.len()), "off": (p.off * 100.0).round() / 100.0})
    };
    if merge {
        let painted = paint_of(&all)?;
        let refs: Vec<(&PathData, FillRule)> = all.iter().map(|(p, r)| (p, *r)).collect();
        let united = po::try_unite_all_with(&refs, po::DEFAULT_PRECISION).map_err(|e| EngineError::Other(e.to_string()))?;
        if united.is_empty() {
            return Err(bad(C, "the shapes unite to nothing"));
        }
        // The merged shape keeps the style of the frontmost object's top shape (its strokes, opacity).
        let style = objects.last().and_then(|(id, _)| {
            let mut leaves = vec![];
            pf_leaves(d, d.node(*id)?, &mut leaves);
            leaves.last().map(style_of)
        });
        let ids: Vec<NodeId> = objects.iter().map(|o| o.0).collect();
        let fill = fill_of(&painted, xf);
        let id = s.edit("Gradient from Image", |d, sel| {
            let mut n = shape_node(d, united, None);
            if let Some(st) = &style {
                st.apply(&mut n);
            }
            n.appearance.set_fill(fill);
            let id = d.wrap(&ids, |_| n)?;
            sel.set([id]);
            Ok(id)
        })?;
        return Ok(json!({"ids": [id.0], "image": image_id.0, "results": [report(id, &painted)]}));
    }
    // Each object its own: paths and compound paths take the fill; a group's shapes each take it.
    let mut fills: Vec<(NodeId, tr::PixelPaint)> = vec![];
    for (id, shapes) in &objects {
        let refs: Vec<&(PathData, FillRule)> = shapes.iter().collect();
        fills.push((*id, paint_of(&refs)?));
    }
    let results: Vec<Value> = fills.iter().map(|(id, p)| report(*id, p)).collect();
    let ids: Vec<NodeId> = fills.iter().map(|f| f.0).collect();
    s.edit("Gradient from Image", |d, sel| {
        for (id, painted) in &fills {
            let fill = fill_of(painted, xf);
            let Some(n) = d.node_mut(*id) else { continue };
            set_fill_deep(n, &fill);
        }
        sel.set(ids.iter().copied());
        Ok(())
    })?;
    Ok(json!({"ids": ids.iter().map(|i| i.0).collect::<Vec<_>>(), "image": image_id.0, "results": results}))
}

/// Gives `n` fill `paint`: a group's paths and compound paths, however deep, each take it.
fn set_fill_deep(n: &mut Node, paint: &Paint) {
    match &mut n.kind {
        NodeKind::Group { children, .. } => {
            for c in children.iter_mut() {
                set_fill_deep(std::sync::Arc::make_mut(c), paint);
            }
        }
        NodeKind::Path { .. } | NodeKind::Compound { .. } | NodeKind::Text(_) => n.appearance.set_fill(paint.clone()),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vectorcraft_color::Paint;

    use crate::Session;

    /// A document with a 100 × 50 image of a left-to-right ramp at the origin, and two rectangles
    /// over its two halves.
    fn setup() -> (Session, u64, u64, u64) {
        use vectorcraft_doc::{ImageBlob, ImageObject, Node, NodeKind};
        let mut s = Session::new();
        s.execute("file.new", &json!({"width": 200, "height": 200})).unwrap();
        let img = vectorcraft_trace::Raster::from_fn(100, 50, |x, _| [(20 + x * 2) as u8, 60, (220 - x * 2) as u8, 255]);
        let png = img.encode_png();
        let image = s
            .edit("img", |d, sel| {
                d.images.insert("img1".into(), ImageBlob::new("image/png", png));
                let id = d.alloc_id();
                let l = d.default_layer();
                let obj = ImageObject {
                    key: "img1".into(),
                    width: 100,
                    height: 50,
                    xf: vectorcraft_geom::Affine::IDENTITY,
                    link: None,
                    placement: Default::default(),
                    pixels: None,
                };
                d.insert(l, 0, Node::new(id, NodeKind::Image(Box::new(obj))))?;
                sel.set([id]);
                Ok(id)
            })
            .unwrap();
        let a = s.execute("shape.rectangle", &json!({"x": 0, "y": 0, "width": 50, "height": 50})).unwrap()["id"].as_u64().unwrap();
        let b = s.execute("shape.rectangle", &json!({"x": 50, "y": 0, "width": 50, "height": 50})).unwrap()["id"].as_u64().unwrap();
        (s, image.0, a, b)
    }

    #[test]
    fn merged_shapes_take_the_images_gradient() {
        let (mut s, image, a, b) = setup();
        let r = s.execute(super::GRADIENT_FROM_IMAGE, &json!({"ids": [a, b], "kind": "linear"})).unwrap();
        assert_eq!(r["image"], json!(image));
        assert_eq!(r["results"][0]["kind"], "linear", "{r}");
        assert!(r["results"][0]["off"].as_f64().unwrap() < 2.0, "{r}");
        let id = vectorcraft_doc::NodeId(r["ids"][0].as_u64().unwrap());
        let st = s.active().unwrap();
        assert!(st.doc.node(vectorcraft_doc::NodeId(a)).is_none(), "the two became one");
        assert!(matches!(st.doc.node(id).unwrap().appearance.fill_paint(), Paint::Gradient(_)));
        // One undo step.
        s.execute("edit.undo", &json!({})).unwrap();
        assert!(s.active().unwrap().doc.node(vectorcraft_doc::NodeId(a)).is_some());
    }

    #[test]
    fn each_shape_its_own_and_errors() {
        let (mut s, _, a, b) = setup();
        let r = s.execute(super::GRADIENT_FROM_IMAGE, &json!({"ids": [a, b], "merge": false, "kind": "radial"})).unwrap();
        assert_eq!(r["results"].as_array().unwrap().len(), 2);
        assert_eq!(r["results"][1]["kind"], "radial");
        assert!(s.execute(super::GRADIENT_FROM_IMAGE, &json!({"ids": [a], "kind": "conic"})).is_err());
        assert!(s.execute(super::GRADIENT_FROM_IMAGE, &json!({"ids": [a], "stops": 99})).is_err());
        // Nothing under a shape far away.
        let c = s.execute("shape.rectangle", &json!({"x": 150, "y": 150, "width": 20, "height": 20})).unwrap()["id"].as_u64().unwrap();
        assert!(s.execute(super::GRADIENT_FROM_IMAGE, &json!({"ids": [c]})).is_err());
    }
}
