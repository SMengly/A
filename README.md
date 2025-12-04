#!/usr/bin/env python3
"""
Generate a Mandelbrot set image (CPU, numpy + Pillow).
Usage: python mandelbrot.py
Adjust WIDTH, HEIGHT, MAX_ITER, and the complex-plane window as desired.
"""
from PIL import Image
import numpy as np

# Image size and iteration limit
WIDTH, HEIGHT = 1200, 800
MAX_ITER = 300

# Complex plane window
RE_START, RE_END = -2.5, 1.0
IM_START, IM_END = -1.2, 1.2

def mandelbrot_image(width, height, max_iter, re_start, re_end, im_start, im_end):
    re = np.linspace(re_start, re_end, width, dtype=np.float64)
    im = np.linspace(im_start, im_end, height, dtype=np.float64)
    X, Y = np.meshgrid(re, im)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    # counts will store iteration of escape; default max_iter means "did not escape"
    counts = np.full(C.shape, max_iter, dtype=np.int32)
    mask = np.full(C.shape, True, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = np.abs(Z) > 2.0
        newly_escaped = escaped & mask
        counts[newly_escaped] = i
        mask &= ~escaped
        # small optimization: break early if all escaped
        if not mask.any():
            break

    # Normalize counts to [0,1]
    norm = counts.astype(np.float32) / max_iter

    # Create an RGB image using a simple color mapping (smooth-ish)
    # map norm -> color with a blue->cyan->yellow->red ramp
    def color_map(v):
        # v in [0,1]
        r = np.clip(3.0 * v - 1.0, 0, 1)
        g = np.clip(3.0 * v - 0.5, 0, 1) - r * 0  # simple ramp
        b = np.clip(1.0 - 3.0 * v, 0, 1)
        return (r, g, b)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        col = color_map(norm)
        # assemble channels
    # faster: compute channels explicitly
    r = (np.clip(3.0 * norm - 1.0, 0, 1) * 255).astype(np.uint8)
    g = (np.clip(3.0 * norm - 0.5, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(1.0 - 3.0 * norm, 0, 1) * 255).astype(np.uint8)

    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b

    return Image.fromarray(rgb, mode="RGB")

def main():
    print("Generating Mandelbrot set...")
    img = mandelbrot_image(WIDTH, HEIGHT, MAX_ITER, RE_START, RE_END, IM_START, IM_END)
    out = "mandelbrot.png"
    img.save(out, "PNG")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
