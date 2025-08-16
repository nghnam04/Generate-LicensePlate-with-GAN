#coding=utf-8
import numpy as np
import cv2
from pathlib import Path
import argparse

def extract(data_path, write_path):
    data_path = Path(data_path)
    write_path = Path(write_path)
    if not write_path.exists():
        write_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {write_path}")

    images_path = sorted(data_path.glob('*'))
    print(f"Found {len(images_path)} images in {data_path}")

    for image_path in images_path:
        print(f"Processing: {image_path}")
        # Đọc ảnh với đường dẫn Unicode
        image_data = np.fromfile(str(image_path), dtype=np.uint8)
        print(f"Read {len(image_data)} bytes from {image_path}")
        image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Cannot decode image {image_path}")
            continue
        print(f"Image shape: {image.shape}")

        edges = cv2.Canny(image, 120, 120)
        print(f"Edges shape: {edges.shape}")

        final_path = write_path / image_path.name
        print(f"Saving to: {final_path}")
        cv2.imencode('.jpg', edges)[1].tofile(str(final_path))
        print(f"Saved: {final_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--A_path', required=True, help='path to store the outline images')
    parser.add_argument('--B_path', required=True, help='path of the original images')
    args = parser.parse_args()

    extract(args.B_path, args.A_path)