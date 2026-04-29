from pathlib import Path
import re
import xml.etree.ElementTree as ET

INPUT_DIR = Path(".")
OUTPUT_DIR = INPUT_DIR / "combined"
GAP = 24  # space between the two SVGs

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def parse_length(value: str | None) -> float | None:
    if not value:
        return None
    m = re.match(r"^\s*([0-9.]+)", str(value))
    return float(m.group(1)) if m else None


def get_svg_size(path: Path) -> tuple[float, float, str | None]:
    tree = ET.parse(path)
    root = tree.getroot()

    width = parse_length(root.get("width"))
    height = parse_length(root.get("height"))
    viewBox = root.get("viewBox")

    if (width is None or height is None) and viewBox:
        parts = re.split(r"[,\s]+", viewBox.strip())
        if len(parts) == 4:
            _, _, vb_w, vb_h = map(float, parts)
            width = width or vb_w
            height = height or vb_h

    if width is None or height is None:
        raise ValueError(f"Could not determine size for {path.name}")

    return width, height, viewBox


def extract_base_name(stem: str) -> tuple[str, str] | None:
    """
    Returns (base_name, kind) where kind is 'volume' or 'price'.
    Accepts names like:
      vev-5500-volume
      vev-5500-price
      velvetfruit-extract-volu
      velvetfruit-extract-pric
    """
    lowered = stem.lower()

    patterns = [
        (r"^(.*?)[-_]volume$", "volume"),
        (r"^(.*?)[-_]price$", "price"),
        (r"^(.*?)[-_]volu$", "volume"),
        (r"^(.*?)[-_]pric$", "price"),
    ]

    for pat, kind in patterns:
        m = re.match(pat, lowered)
        if m:
            base = stem[: len(m.group(1))]
            return base.rstrip("-_"), kind

    return None


def build_combined_svg(volume_path: Path, price_path: Path, out_path: Path) -> None:
    vw, vh, v_viewbox = get_svg_size(volume_path)
    pw, ph, p_viewbox = get_svg_size(price_path)

    total_w = vw + GAP + pw
    total_h = max(vh, ph)
    
    SVG_NS = "http://www.w3.org/2000/svg"
    ET.register_namespace("", SVG_NS)
    
    outer = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": str(total_w),
            "height": str(total_h),
            "viewBox": f"0 0 {total_w} {total_h}",
        },
    )

    def add_inner_svg(src_path: Path, x: float, y: float, w: float, h: float, viewbox: str | None):
        tree = ET.parse(src_path)
        root = tree.getroot()

        inner_attrs = {
            "x": str(x),
            "y": str(y),
            "width": str(w),
            "height": str(h),
        }
        if viewbox:
            inner_attrs["viewBox"] = viewbox

        inner = ET.SubElement(outer, f"{{{SVG_NS}}}svg", inner_attrs)

        # Copy everything inside the source root into the nested SVG
        for child in list(root):
            inner.append(child)

        # Copy root-level presentation attrs that matter
        for attr in [
            "preserveAspectRatio",
            "style",
            "class",
        ]:
            if root.get(attr) is not None:
                inner.set(attr, root.get(attr))

    # Vertically center both SVGs
    add_inner_svg(volume_path, 0, (total_h - vh) / 2, vw, vh, v_viewbox)
    add_inner_svg(price_path, vw + GAP, (total_h - ph) / 2, pw, ph, p_viewbox)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(outer).write(out_path, encoding="utf-8", xml_declaration=True)


def main():
    files = sorted(INPUT_DIR.glob("*.svg"))
    pairs: dict[str, dict[str, Path]] = {}

    for f in files:
        if f.parent == OUTPUT_DIR:
            continue

        parsed = extract_base_name(f.stem)
        if not parsed:
            continue

        base, kind = parsed
        pairs.setdefault(base, {})[kind] = f

    for base, group in pairs.items():
        vol = group.get("volume")
        price = group.get("price")
        if not vol or not price:
            print(f"Skipping {base}: missing volume or price file")
            continue

        out_file = OUTPUT_DIR / f"{base}_info.svg"
        if out_file.exists():
            print(f"Skipping {out_file.name}: already exists")
            continue

        build_combined_svg(vol, price, out_file)
        print(f"Created {out_file}")

if __name__ == "__main__":
    main()
