#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path('.')
FILES = [
    ('sigma = 0.00, alpha_{S} = 0.17', BASE / 'combo_sigma0p00_alpha0p17.png'),
    ('sigma = 0.50, alpha_{S} = 0.17', BASE / 'combo_sigma0p50_alpha0p17.png'),
    ('sigma = 0.17, alpha_{S} = 0.09', BASE / 'combo_sigma0p17_alpha0p09.png'),
    ('sigma = 0.00, alpha_{S} = 0.06', BASE / 'combo_sigma0p00_alpha0p06.png'),
]

OUT_PNG = BASE / 'combo_four_2x2_final.png'
OUT_PDF = BASE / 'combo_four_2x2_final.pdf'

panel_w = 1050
title_h = 70
panel_pad = 24
outer_pad = 28
bg = 'white'
title_color = 'black'


def load_font(size: int):
    candidates = [
        '/System/Library/Fonts/Times.ttc',
        '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
        '/System/Library/Fonts/Supplemental/Georgia.ttf',
        '/Library/Fonts/Times New Roman.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


title_font = load_font(34)


def fit_image(img: Image.Image, target_w: int) -> Image.Image:
    scale = target_w / img.width
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)


panels = []
max_panel_h = 0

for title, path in FILES:
    if not path.exists():
        raise FileNotFoundError(f'Missing image: {path}')

    img = Image.open(path).convert('RGB')
    img = fit_image(img, panel_w)

    panel = Image.new('RGB', (panel_w, title_h + img.height), bg)
    draw = ImageDraw.Draw(panel)

    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (panel_w - tw) // 2
    ty = (title_h - th) // 2 - 3

    draw.text((tx, ty), title, fill=title_color, font=title_font)
    panel.paste(img, (0, title_h))

    panels.append(panel)
    max_panel_h = max(max_panel_h, panel.height)

norm_panels = []
for p in panels:
    if p.height < max_panel_h:
        canvas = Image.new('RGB', (panel_w, max_panel_h), bg)
        canvas.paste(p, (0, 0))
        norm_panels.append(canvas)
    else:
        norm_panels.append(p)

canvas_w = outer_pad * 2 + panel_w * 2 + panel_pad
canvas_h = outer_pad * 2 + max_panel_h * 2 + panel_pad
canvas = Image.new('RGB', (canvas_w, canvas_h), bg)

positions = [
    (outer_pad, outer_pad),
    (outer_pad + panel_w + panel_pad, outer_pad),
    (outer_pad, outer_pad + max_panel_h + panel_pad),
    (outer_pad + panel_w + panel_pad, outer_pad + max_panel_h + panel_pad),
]

for p, pos in zip(norm_panels, positions):
    canvas.paste(p, pos)

canvas.save(OUT_PNG)
canvas.save(OUT_PDF, 'PDF', resolution=300.0)

print(f'Saved {OUT_PNG}')
print(f'Saved {OUT_PDF}')
