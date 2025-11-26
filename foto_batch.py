import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

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

def make_idphoto(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Igazolványkép: középre vágás + fekete-fehér konverzió.
    """
    out = ImageOps.fit(
        img,
        (target_w, target_h),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )

    # Fekete-fehér átalakítás
    out = out.convert("L").convert("RGB")
    return out


def detect_face_and_crop(img: Image.Image) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        print("⚠ Nincs arc felismerve — eredeti kép marad.")
        return img

    (x, y, w, h) = max(faces, key=lambda box: box[2] * box[3])

    pad = int(w * 0.5)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(cv_img.shape[1], x + w + pad)
    y2 = min(cv_img.shape[0], y + h + pad)

    cropped = cv_img[y1:y2, x1:x2]

    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


def main():
    parser = argparse.ArgumentParser(description="Batch képfeldolgozó")
    parser.add_argument("input", type=str, help="Bemeneti mappa vagy kép")
    parser.add_argument("-o", "--output", type=str, default="output", help="Kimeneti mappa")
    parser.add_argument("--idphoto", type=str, help="Igazolványkép méret px-ben, pl.: 413x531")

    args = parser.parse_args()
    use_face = input("\nSzeretnél arc detektálást használni? (y/n): ").strip().lower() == "y"


    from datetime import datetime

    # automatikus mappa
    if args.idphoto:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        outdir = Path(f"igkep-{timestamp}")
    else:
        outdir = Path(args.output)

    outdir.mkdir(exist_ok=True)

    inpath = Path(args.input)
    files = list(iter_images(inpath))

    if not files:
        print("Nincsenek képek a megadott mappában.")
        return

    # 1. LÉPÉS – Alap vagy igazolványkép feldolgozás
    for f in files:
        try:
            with Image.open(f) as img:
                img = auto_orient(img)

                if args.idphoto:
                    try:
                        w, h = map(int, args.idphoto.lower().split("x"))
                        img = make_idphoto(img, w, h)
                    except:
                        print("Hibás idphoto paraméter.")

                outpath = outdir / f.name
                img.save(outpath)
                print(f"OK: {f} -> {outpath}")

        except Exception as e:
            print(f"Hiba: {e}")

    # 2. LÉPÉS – Arc detektálás kérdés
    answer = input("\nSzeretnél arc detektálást alkalmazni a képekre? (y/n): ").strip().lower()

    if answer == "y":
        print("\nArc-detektálás fut...\n")

    for f in files:
        try:
            with Image.open(f) as img:
                img = auto_orient(img)

                # 1) ARC DETEKTÁLÁS, ha kérték
                if use_face:
                    img = detect_face_and_crop(img)

                # 2) IGAZOLVÁNY MÉRET, ha kérték
                if args.idphoto:
                    try:
                        w, h = map(int, args.idphoto.lower().split("x"))
                        img = make_idphoto(img, w, h)
                    except:
                        print("Hibás idphoto paraméter.")

                # 3) FÁJLNÉV – ha arc detektálás volt, legyen más neve
                if use_face:
                    outpath = outdir / f"face_{f.name}"
                else:
                    outpath = outdir / f.name

                # 4) MENTÉS
                img.save(outpath)
                print(f"OK: {f} -> {outpath}")

        except Exception as e:
            print(f"Hiba: {e}")


        else:
            print("Arc-detektálás kihagyva.")


if __name__ == "__main__":
    main()
