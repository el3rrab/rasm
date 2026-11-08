//! Rasm, our own add-on, in the interface (Object › Plug-ins › Rasm). Gradient from Image… opens a
//! dialog that picks the kind of gradient (Auto, Linear, Radial), whether the shapes are merged into
//! one, the most stops and how smooth the ramps are, with Preview on the canvas; OK runs
//! `addon.rasm.gradientFromImage` on the shapes selected when the dialog opened.

use egui::Ui;
use serde_json::{Value, json};

use crate::VectorcraftApp;
use crate::dialogs::{DialogSpec, form};
use crate::menus::Item;
use crate::state::Dialog;
use crate::theme::Tokens;
use crate::widgets;

/// The command OK runs, and the dialog's kind.
pub const COMMAND: &str = "addon.rasm.gradientFromImage";
pub const KIND: &str = "rasmGradientFromImage";
const LABEL_W: f32 = 110.0;
const FIELD_W: f32 = 64.0;

/// Object › Plug-ins › Rasm.
pub fn menu() -> Vec<Item> {
    vec![Item::Cmd("Gradient from Image…", COMMAND, Value::Null)]
}

/// The fields the dialog opens with: the selected objects, Auto, merged, 8 stops, smoothness 3,
/// Preview on.
pub(crate) fn fields(app: &VectorcraftApp) -> Value {
    let ids: Vec<u64> = app.session.active().map_or(vec![], |st| st.selection.objects.iter().map(|i| i.0).collect());
    json!({ "ids": ids, "kind": "auto", "merge": true, "stops": 8, "smoothness": 3.0, "preview": true })
}

/// The command's parameters from the dialog's fields.
fn params(d: &Dialog) -> Value {
    json!({
        "ids": d.fields.get("ids").cloned().unwrap_or(json!([])),
        "kind": d.str("kind"),
        "merge": d.bool("merge"),
        "stops": d.f64("stops", 8.0).round().clamp(2.0, 16.0),
        "smoothness": d.f64("smoothness", 3.0).clamp(0.5, 32.0),
    })
}

pub(crate) const SPEC: DialogSpec = DialogSpec {
    heading: |_| tl!("Gradient from Image").to_string(),
    body,
    confirm,
    min_width: 360.0,
    ok_enabled: Some(|_, d| d.fields.get("ids").and_then(Value::as_array).is_some_and(|a| !a.is_empty())),
    ..DialogSpec::FORM
};

fn body(app: &mut VectorcraftApp, ui: &mut Ui, d: &mut Dialog) -> bool {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(tl!("Fills the selected shapes with the colors of the image under them.")).color(t.text_dim));
    ui.add_space(6.0);
    let kind = d.str("kind");
    widgets::label_row(ui, tl!("Gradient:"), LABEL_W, |ui| {
        for (value, label) in [("auto", "Auto"), ("linear", "Linear"), ("radial", "Radial")] {
            if widgets::radio(ui, label, kind == value, true) {
                d.fields.insert("kind".into(), json!(value));
            }
        }
    });
    widgets::label_row(ui, tl!("Most Stops:"), LABEL_W, |ui| {
        if let Some(v) = widgets::range_field(ui, "rasm-stops", d.f64("stops", 8.0), 2.0..=16.0, "", 0, FIELD_W) {
            d.fields.insert("stops".into(), json!(v));
        }
    });
    widgets::label_row(ui, tl!("Smoothness:"), LABEL_W, |ui| {
        if let Some(v) = widgets::range_field(ui, "rasm-smoothness", d.f64("smoothness", 3.0), 0.5..=32.0, "", 1, FIELD_W) {
            d.fields.insert("smoothness".into(), json!(v));
        }
        ui.label(egui::RichText::new(tl!("more is fewer stops")).color(t.text_dim));
    });
    ui.add_space(4.0);
    form::check(ui, d, "merge", tl!("Merge into One Shape"));
    if d.fields.get("ids").and_then(Value::as_array).is_none_or(Vec::is_empty) {
        ui.add_space(6.0);
        ui.label(egui::RichText::new(tl!("Select the shapes to fill, over the image they were traced from.")).color(t.text_dim));
        return false;
    }
    form::preview(app, ui, d, "Gradient from Image", COMMAND, params(d));
    false
}

/// OK: keep the previewed result (or run it) as one undo step and close.
fn confirm(app: &mut VectorcraftApp, d: &Dialog) -> Result<Value, String> {
    form::commit_preview(app, COMMAND, params(d))
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vectorcraft_color::Paint;
    use vectorcraft_engine::Session;

    use crate::tests_labels::painted_text;
    use crate::{VectorcraftApp, menus};

    #[test]
    fn the_dialog_shows_its_controls_and_ok_fills_the_shapes() {
        let mut app = VectorcraftApp::new(Session::new(), Default::default());
        app.run("file.new", json!({"width": 200, "height": 200})).unwrap();
        // The image: a ramp, placed from a PNG file.
        let img = image::RgbaImage::from_fn(100, 50, |x, _| image::Rgba([(20 + x * 2) as u8, 60, (220 - x * 2) as u8, 255]));
        let path = std::env::temp_dir().join(format!("rasm-ramp-{}.png", std::process::id()));
        img.save(&path).unwrap();
        app.run("file.place", json!({"path": path.to_string_lossy(), "x": 0, "y": 0})).unwrap();
        let _ = std::fs::remove_file(&path);
        // Two shapes over the image, wherever it was placed.
        let b = {
            let st = app.session.active().unwrap();
            st.doc.bounds_of(&st.selection.objects, false).unwrap()
        };
        let (w, h) = (b.width() / 2.0, b.height());
        app.run("shape.rectangle", json!({"x": b.x0, "y": b.y0, "width": w, "height": h})).unwrap();
        app.run("shape.rectangle", json!({"x": b.x0 + w, "y": b.y0, "width": w, "height": h})).unwrap();
        app.run("select.all", json!({})).unwrap();
        let entries = menus::menu_entries(&app);
        let item = entries.iter().find(|e| e.path == ["Object", "Plug-ins", "Rasm"] && e.label == "Gradient from Image…").unwrap();
        let (cmd, p) = menus::click_target(&item.label, item.command.as_deref().unwrap(), &item.params);
        menus::invoke(&mut app, &cmd, p);
        assert_eq!(app.ui.dialog.as_ref().map(|d| d.kind.as_str()), Some(super::KIND));
        let shown = painted_text(&mut app, |app, ui| crate::dialogs::show(app, ui.ctx()));
        for s in ["Gradient from Image", "Gradient:", "Auto", "Linear", "Radial", "Most Stops:", "Smoothness:", "Merge into One Shape", "Preview"] {
            assert!(shown.contains(s), "{s} in {shown}");
        }
        crate::dialogs::confirm(&mut app).unwrap();
        assert!(app.ui.dialog.is_none());
        let st = app.session.active().unwrap();
        let id = st.selection.objects[0];
        assert!(matches!(st.doc.node(id).unwrap().appearance.fill_paint(), Paint::Gradient(_)), "filled with a gradient");
    }
}
