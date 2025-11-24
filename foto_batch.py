import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(input_path: Path):
    """Bemenet: fájl vagy mappa. Kimenet: képek listája."""
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTS:
            yield input_path
        return
    for p in input_path.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXTS:
            yield p

def auto_orient(img: Image.Image) -> Image.Image:
    """EXIF alapján automatikus tájolás."""
    try:
        return ImageOps.exif_transpose(img)
    except:
        return img

def main():
    parser = argparse.ArgumentParser(description="Batch képfeldolgozó")
    parser.add_argument("input", type=str, help="Bemeneti mappa vagy kép")
    parser.add_argument("-o", "--output", type=str, default="output", help="Kimeneti mappa")

    args = parser.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.output)
    outdir.mkdir(exist_ok=True)

    files = list(iter_images(inpath))
    if not files:
        print("Nincsenek képek a megadott mappában.")
        return

    for f in files:
        try:
            with Image.open(f) as img:
                img = auto_orient(img)
                outpath = outdir / f.name
                img.save(outpath)
                print(f"OK: {f} -> {outpath}")
        except Exception as e:
            print(f"Hiba: {e}")

if __name__ == "__main__":
    main()
