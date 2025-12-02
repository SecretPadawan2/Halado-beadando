import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTS:
            yield input_path
        return
    for p in input_path.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def auto_orient(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except:
        return img


def detect_face_and_crop(img: Image.Image) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        print("Nincs arc felismerve — kép változatlan marad.")
        return img

    (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])
    pad = int(w * 0.5)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(cv_img.shape[1], x + w + pad)
    y2 = min(cv_img.shape[0], y + h + pad)

    cropped = cv_img[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


def make_idphoto(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    out = ImageOps.fit(
        img,
        (target_w, target_h),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )
    return out.convert("L").convert("RGB")


def apply_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


def apply_rotate(img: Image.Image, angle: int) -> Image.Image:
    return img.rotate(angle, expand=True)


def apply_blur(img: Image.Image, amount: int) -> Image.Image:
    arr = cv2.GaussianBlur(np.array(img), (amount * 2 + 1, amount * 2 + 1), 0)
    return Image.fromarray(arr)


def apply_sharp(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.SHARPEN)


def apply_watermark(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(255, 255, 255))
    return img


def main():
    parser = argparse.ArgumentParser(description="Moduláris batch képfeldolgozó")

    parser.add_argument("input", type=str, help="Bemeneti mappa vagy kép")

    # Funkciók:
    parser.add_argument("--face", action="store_true", help="Arc detektálás automatikusan")
    parser.add_argument("--idphoto", type=str, help="Igazolványkép méret, pl. 413x531")
    parser.add_argument("--grayscale", action="store_true", help="Fekete-fehér")
    parser.add_argument("--rotate", type=int, help="Elforgatás fokban")
    parser.add_argument("--blur", type=int, help="Gauss blur pixelekben")
    parser.add_argument("--sharp", action="store_true", help="Élesítés")
    parser.add_argument("--watermark", type=str, help="Vízjel szöveg")

    args = parser.parse_args()

    if args.face:
        use_face = True
        print("Arc detektálás: BE (parancssorból)")
    else:
        ask = input("Szeretnél arc detektálást használni? (y/n): ").strip().lower()
        use_face = (ask == "y")
        print("Arc detektálás:", "BE" if use_face else "KI")

    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")

    used = []

    if use_face:
        used.append("face")

    if args.idphoto:
        used.append("idphoto")

    if args.grayscale:
        used.append("grayscale")

    if args.rotate:
        used.append(f"rotate{args.rotate}")

    if args.blur:
        used.append(f"blur{args.blur}")

    if args.sharp:
        used.append("sharp")

    if args.watermark:
        used.append("watermark")

    # Ha semmi extra funkció nincs → alap
    if not used:
        used.append("alap")

    # Mappa felépítése
    folder_name = "-".join(used) + "-" + timestamp

    outdir = Path(folder_name)

    outdir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------

    inpath = Path(args.input)
    files = list(iter_images(inpath))

    if not files:
        print("Nincsenek képek a megadott helyen.")
        return

    print("\n--- Feldolgozás indul ---\n")

    for f in files:
        try:
            with Image.open(f) as img:
                img = auto_orient(img)

                # Pipeline sorrendben:
                if use_face:
                    img = detect_face_and_crop(img)

                if args.idphoto:
                    w, h = map(int, args.idphoto.lower().split("x"))
                    img = make_idphoto(img, w, h)

                if args.grayscale:
                    img = apply_grayscale(img)

                if args.rotate:
                    img = apply_rotate(img, args.rotate)

                if args.blur:
                    img = apply_blur(img, args.blur)

                if args.sharp:
                    img = apply_sharp(img)

                if args.watermark:
                    img = apply_watermark(img, args.watermark)

                outpath = outdir / f.name
                img.save(outpath)
                print(f"OK: {f.name} -> {outpath}")

        except Exception as e:
            print(f"Hiba a(z) {f.name} feldolgozásakor: {e}")

    print("\nKész!\n")


if __name__ == "__main__":
    main()
