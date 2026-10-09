"""Render the Hampa 40s 9:16 motion-graphics ad (visuals + SFX bed).

Usage: python3 render.py <out_dir>
Writes <out_dir>/video.mp4 (silent) and <out_dir>/sfx.wav.
Cue times are absolute seconds, synced to the voice takes (voice starts
0.25s into each 10s scene).
"""
import math, os, random, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
W, H, FPS, DUR = 1080, 1920, 30, 40.0
SR = 48000

PAPER = (243, 235, 221)
INK = (22, 19, 16)
AMBER = (181, 101, 29)
AMBER_D = (122, 62, 14)
AMBER_L = (236, 160, 64)
GREEN = (62, 142, 65)
GREEN_D = (24, 82, 40)
GREY = (150, 150, 140)
WHITE = (255, 255, 255)

F_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
if not os.path.exists(F_BLACK):
    F_BLACK = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
F_BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"

random.seed(7)
rng = np.random.default_rng(7)


# ---------- easing / helpers ----------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def prog(t, t0, d):
    return clamp((t - t0) / d)

def ease_out(x):
    return 1 - (1 - x) ** 3

def ease_in(x):
    return x ** 3

def back(x, s=2.2):
    x -= 1
    return x * x * ((s + 1) * x + s) + 1

_font_cache = {}
def font(path, size):
    k = (path, size)
    if k not in _font_cache:
        _font_cache[k] = ImageFont.truetype(path, size)
    return _font_cache[k]

_cache = {}
def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]

def place(canvas, img, cx, cy, scale=1.0, rot=0.0, alpha=1.0):
    if alpha <= 0.01 or scale <= 0.01:
        return
    im = img
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if abs(rot) > 0.05:
        im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 0.999:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im = im.copy(); im.putalpha(a)
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


# ---------- assets ----------
def paper_bg(color=PAPER, dots=True, seed=0):
    r = np.random.default_rng(seed)
    base = np.zeros((H, W, 3), np.float32) + np.array(color, np.float32)
    grain = r.normal(0, 6, (H // 2, W // 2)).astype(np.float32)
    grain = np.kron(grain, np.ones((2, 2), np.float32))[:H, :W]
    base += grain[:, :, None]
    # soft vignette
    yy, xx = np.mgrid[0:H, 0:W]
    v = ((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2
    base *= (1 - 0.35 * v)[:, :, None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    if dots:
        d = ImageDraw.Draw(img)
        dc = tuple(int(c * 0.86) for c in color) + (255,)
        for y in range(0, 520, 26):
            for x in range(W - 520, W, 26):
                fx = (x - (W - 520)) / 520
                fy = 1 - y / 520
                rad = 9 * fx * fy
                if rad > 0.8:
                    d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=dc)
    return img

def torn_rect(w, h, color, seed=0, jag=10):
    r = random.Random(seed)
    pts = []
    step = 18
    for x in range(0, w + 1, step):
        pts.append((x, r.uniform(0, jag)))
    for y in range(0, h + 1, step):
        pts.append((w - r.uniform(0, jag), y))
    for x in range(w, -1, -step):
        pts.append((x, h - r.uniform(0, jag)))
    for y in range(h, -1, -step):
        pts.append((r.uniform(0, jag), y))
    pad = 30
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).polygon([(x + pad + 8, y + pad + 10) for x, y in pts], fill=(0, 0, 0, 90))
    sh = sh.filter(ImageFilter.GaussianBlur(8))
    im.alpha_composite(sh)
    ImageDraw.Draw(im).polygon([(x + pad, y + pad) for x, y in pts], fill=color + (255,))
    return im

def text_img(txt, size, color=INK, path=F_BLACK, bg=None, padx=34, pady=18, seed=0, tape=False):
    f = font(path, size)
    bb = f.getbbox(txt)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    if bg is None:
        im = Image.new("RGBA", (tw + 20, th + 20), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10 - bb[0], 10 - bb[1]), txt, font=f, fill=color + (255,))
        return im
    block = torn_rect(tw + padx * 2, th + pady * 2, bg, seed=seed, jag=7)
    ImageDraw.Draw(block).text((30 + padx - bb[0], 30 + pady - bb[1]), txt, font=f, fill=color + (255,))
    if tape:
        t = Image.new("RGBA", (120, 40), (245, 240, 220, 170))
        t = t.rotate(-8, expand=True)
        block.alpha_composite(t, (10, 4))
    return block

def drop_img(size):
    """glowing amber oil droplet (teardrop)."""
    s = size
    im = Image.new("RGBA", (s * 2, int(s * 2.6)), (0, 0, 0, 0))
    cx, cy = s, int(s * 1.55)
    glow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([cx - s * 0.9, cy - s * 0.9, cx + s * 0.9, cy + s * 0.9], fill=AMBER_L + (150,))
    glow = glow.filter(ImageFilter.GaussianBlur(s * 0.35))
    im.alpha_composite(glow)
    d = ImageDraw.Draw(im)
    r = s * 0.55
    d.polygon([(cx, cy - r * 2.1), (cx - r * 0.93, cy - r * 0.35), (cx + r * 0.93, cy - r * 0.35)], fill=AMBER + (255,))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=AMBER + (255,))
    # inner gradient
    for i in range(10):
        k = i / 10
        rr = r * (1 - k * 0.75)
        c = tuple(int(AMBER[j] + (AMBER_L[j] - AMBER[j]) * k) for j in range(3))
        d.ellipse([cx - rr + r * 0.12 * k, cy - rr + r * 0.1 * k, cx + rr + r * 0.12 * k, cy + rr + r * 0.1 * k], fill=c + (255,))
    d.ellipse([cx - r * 0.5, cy - r * 0.75, cx - r * 0.15, cy - r * 0.3], fill=(255, 240, 210, 220))
    return im

def person_img(color, seed):
    r = random.Random(seed)
    im = Image.new("RGBA", (70, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([12, 44, 58, 116], radius=18, fill=color + (255,))
    d.ellipse([20, 6, 50, 38], fill=color + (255,))
    # white paper border look
    bord = im.filter(ImageFilter.MaxFilter(7))
    a = bord.getchannel("A")
    base = Image.new("RGBA", im.size, (255, 255, 255, 0)); base.putalpha(a)
    w = Image.new("RGBA", im.size, (255, 253, 245, 255)); w.putalpha(a)
    w.alpha_composite(im)
    return w

def tin_img(scale=1.0, greasy=True):
    w, h = int(300 * scale), int(190 * scale)
    im = Image.new("RGBA", (w + 40, h + 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 50, 20 + w, 30 + h], fill=(120, 124, 118, 255))
    d.ellipse([20, 30 + h - 40, 20 + w, 30 + h + 20], fill=(100, 104, 98, 255))
    d.ellipse([20, 20, 20 + w, 80], fill=(175, 178, 168, 255))
    if greasy:
        d.ellipse([60, 32, 20 + w - 50, 68], fill=(205, 205, 190, 255))
        for i in range(5):
            x = 40 + i * w // 5
            d.line([(x, 60), (x + 8, 60 + h * 0.6)], fill=(190, 192, 180, 160), width=6)
    bord = im.getchannel("A").filter(ImageFilter.MaxFilter(11))
    out = Image.new("RGBA", im.size, (255, 253, 245, 0)); out.putalpha(bord)
    wb = Image.new("RGBA", im.size, (255, 253, 245, 255)); wb.putalpha(bord)
    wb.alpha_composite(im)
    return wb

def leaf_img(size, color=GREEN):
    """stylised serrated 7-finger leaf, heavily blurred (subliminal)."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = size / 2, size * 0.78
    lens = [0.32, 0.42, 0.5, 0.56, 0.5, 0.42, 0.32]
    angs = [-80, -52, -26, 0, 26, 52, 80]
    for L, a in zip(lens, angs):
        a = math.radians(a - 90)
        L *= size
        wid = L * 0.13
        tip = (cx + math.cos(a) * L, cy + math.sin(a) * L)
        nx, ny = -math.sin(a), math.cos(a)
        mid = (cx + math.cos(a) * L * 0.5, cy + math.sin(a) * L * 0.5)
        d.polygon([(cx, cy), (mid[0] + nx * wid, mid[1] + ny * wid), tip, (mid[0] - nx * wid, mid[1] - ny * wid)], fill=color + (255,))
    d.line([(cx, cy), (cx, cy + size * 0.18)], fill=color + (255,), width=int(size * 0.015))
    return im.filter(ImageFilter.GaussianBlur(size * 0.045))

def product_img():
    p = Image.open(os.path.join(HERE, "assets/product_cutout.png")).convert("RGBA")
    a = p.getchannel("A").filter(ImageFilter.MaxFilter(17))
    out = Image.new("RGBA", (p.width + 60, p.height + 60), (0, 0, 0, 0))
    sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
    shadow = Image.new("RGBA", p.size, (0, 0, 0, 110)); shadow.putalpha(a.point(lambda v: v * 110 // 255))
    sh.alpha_composite(shadow, (42, 46)); sh = sh.filter(ImageFilter.GaussianBlur(12))
    out.alpha_composite(sh)
    wb = Image.new("RGBA", p.size, (255, 253, 245, 255)); wb.putalpha(a)
    out.alpha_composite(wb, (30, 30)); out.alpha_composite(p, (30, 30))
    return out

def marker_circle(canvas, cx, cy, rx, ry, p, width=12, color=INK, seed=0):
    """hand-drawn ellipse drawing itself, p in 0..1"""
    if p <= 0:
        return
    r = random.Random(seed)
    jit = [r.uniform(-0.06, 0.06) for _ in range(8)]
    n = int(80 * p) + 2
    pts = []
    for i in range(n):
        a = -math.pi / 2 + 2.15 * math.pi * (i / 80)
        k = 1 + jit[int((i / 80) * 7) % 8] * math.sin(a * 3)
        pts.append((cx + math.cos(a) * rx * k, cy + math.sin(a) * ry * k))
    ImageDraw.Draw(canvas).line(pts, fill=color + (255,), width=width, joint="curve")

def tick(canvas, x, y, s, p, color=INK, width=16):
    if p <= 0:
        return
    a = (x, y); b = (x + s * 0.35, y + s * 0.35); c = (x + s, y - s * 0.55)
    d = ImageDraw.Draw(canvas)
    if p < 0.4:
        k = p / 0.4
        d.line([a, (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)], fill=color + (255,), width=width)
    else:
        k = (p - 0.4) / 0.6
        d.line([a, b, (b[0] + (c[0] - b[0]) * k, b[1] + (c[1] - b[1]) * k)], fill=color + (255,), width=width, joint="curve")

def crackle(canvas, cx, cy, p, seed):
    r = random.Random(seed + int(p * 6))
    d = ImageDraw.Draw(canvas)
    for k in range(5):
        a = r.uniform(0, 2 * math.pi)
        pts = [(cx, cy)]
        x, y = cx, cy
        for _ in range(4):
            x += math.cos(a) * r.uniform(30, 60) + r.uniform(-20, 20)
            y += math.sin(a) * r.uniform(30, 60) + r.uniform(-20, 20)
            pts.append((x, y))
        d.line(pts, fill=INK + (255,), width=9)


# ---------- persistent assets ----------
BG = paper_bg(PAPER, seed=1)
BG_GREEN = paper_bg((226, 236, 218), seed=2)
BG_AMBER = paper_bg((240, 214, 170), seed=3)
DROP = drop_img(150)
PRODUCT = product_img()
LEAF = leaf_img(900, (86, 160, 80))
PEOPLE = [person_img(c, i) for i, c in enumerate([INK, GREEN_D, AMBER_D, (70, 60, 52), GREEN])]
CROWD = [(90 + (i % 9) * 112 + (14 if (i // 9) % 2 else 0), 760 + (i // 9) * 150, random.Random(i).randrange(5)) for i in range(9 * 6)]


# ---------- scenes ----------
def scene1(c, t):
    # drop falls
    p = prog(t, 0.0, 0.6)
    if t < 1.2:
        place(c, DROP, W / 2, -200 + ease_in(p) * 520, 0.8)
    # amber block + headline slam
    s = back(prog(t, 0.25, 0.35))
    place(c, cached("s1blk", lambda: torn_rect(940, 430, AMBER, seed=3)), W / 2, 420, s, -2)
    place(c, cached("s1a", lambda: text_img("NEARLY", 70, WHITE, F_BOLD)), W / 2, 300, s, -2)
    place(c, cached("s1b", lambda: text_img("20 CRORE", 160, WHITE)), W / 2, 445, s, -2)
    sub = back(prog(t, 1.2, 0.3))
    place(c, cached("s1c", lambda: text_img("Indians live with joint pain", 52, INK, F_BOLD, bg=PAPER, seed=4, tape=True)), W / 2, 640, sub, 1.5)
    # crowd multiplies
    for i, (x, y, k) in enumerate(CROWD):
        ti = 1.0 + i * 0.05
        if t > ti:
            sc = back(prog(t, ti, 0.25))
            dim = 0.55 if t > 4.6 else 1.0
            place(c, PEOPLE[k], x, y, sc * 1.05, 0, dim)
    for j, idx in enumerate([3, 11, 16, 22, 29, 34, 40, 47]):
        x, y, _ = CROWD[idx]
        marker_circle(c, x, y + 30, 58, 76, prog(t, 2.6 + j * 0.12, 0.3), 8, (200, 40, 30), seed=idx)
    # balm tin slap + question
    if t > 4.55:
        sc = back(prog(t, 4.55, 0.25), 3)
        place(c, cached("tin", lambda: tin_img(1.6)), W / 2, 1180, sc, -6)
        q = back(prog(t, 4.9, 0.3))
        place(c, cached("s1q", lambda: text_img("Still stuck on sticky balm?", 64, WHITE, F_BLACK, bg=INK, seed=5)), W / 2, 1520, q, -2)

def scene2(c, t):
    beats = [(0.25, "Stiff mornings."), (1.79, "Steeper stairs."), (3.83, "Smell for hours."), (5.73, "Relief gone by lunch.")]
    # sunrise torn paper
    place(c, cached("sun", lambda: torn_rect(420, 420, AMBER_L, seed=8, jag=20)), 820, 330, ease_out(prog(t, 0, 0.5)) * 0.9, 12)
    # beat 1: stiff knee figure + crackle
    if t < 7.6:
        fig = cached("fig", lambda: person_img(INK, 99).resize((210, 360)))
        place(c, fig, 280, 760, back(prog(t, 0.1, 0.3)))
        if t > 0.4:
            crackle(c, 300, 860, t, 11)
    # beat 2: stairs stretch
    if t > 1.7 and t < 7.6:
        g = ease_out(prog(t, 1.79, 1.6))
        d = ImageDraw.Draw(c)
        for i in range(6):
            sh = 60 + 70 * g
            x0 = 560 + i * 80
            y0 = 1150 - i * sh
            d.rectangle([x0, y0, 1080, y0 + sh], fill=GREEN_D + (255,), outline=PAPER + (255,), width=4)
        place(c, cached("fig2", lambda: person_img(AMBER_D, 5).resize((120, 200))), 620, 1150 - 70 * g - 110, back(prog(t, 1.8, 0.3)))
    # beat 3: tin + smell lines + draining bar
    if t > 3.7 and t < 7.75:
        place(c, cached("tin2", lambda: tin_img(1.2)), 330, 1450, back(prog(t, 3.83, 0.3)))
        d = ImageDraw.Draw(c)
        for k in range(3):
            pts = [(260 + k * 70 + 25 * math.sin(t * 6 + y / 40 + k), 1310 - y) for y in range(0, int(260 * prog(t, 3.9, 0.6)), 10)]
            if len(pts) > 1:
                d.line(pts, fill=(130, 140, 110, 255), width=10, joint="curve")
        bar = 1 - ease_out(prog(t, 5.8, 1.5))
        d.rectangle([640, 1360, 760, 1760], outline=INK + (255,), width=8)
        d.rectangle([652, 1748 - 376 * bar, 748, 1748], fill=GREEN + (255,))
    # beat captions
    for i, (bt, txt) in enumerate(beats):
        if bt <= t < 7.6:
            place(c, cached("b%d" % i, lambda txt=txt, i=i: text_img(txt, 66, WHITE if i % 2 == 0 else INK, F_BLACK, bg=INK if i % 2 == 0 else PAPER, seed=20 + i, tape=i % 2 == 1)),
                  W / 2, 180 + i * 0, back(prog(t, bt, 0.25)), -2 + i) if i == max(j for j, (b, _) in enumerate(beats) if b <= t) else None
    # AB BAS: drop smash
    if t > 7.2:
        p = prog(t, 7.2, 0.4)
        if p < 1:
            place(c, DROP, 330, -150 + ease_in(p) * 1550, 1.0)
        if t > 7.6:
            k = prog(t, 7.6, 0.5)
            r = random.Random(3)
            d = ImageDraw.Draw(c)
            for j in range(18):
                a = r.uniform(0, 2 * math.pi); dist = 900 * ease_out(k) * r.uniform(0.4, 1)
                x, y = 330 + math.cos(a) * dist, 1400 + math.sin(a) * dist
                sz = r.uniform(20, 60)
                d.polygon([(x, y), (x + sz, y + sz * 0.3), (x + sz * 0.4, y + sz)], fill=(150, 154, 146, 255))
            blk = cached("abbas_blk", lambda: torn_rect(1000, 560, AMBER, seed=31, jag=16))
            sc = back(prog(t, 7.62, 0.3), 3)
            place(c, blk, W / 2, 960, sc, -3)
            place(c, cached("abbas", lambda: text_img("AB BAS.", 185, WHITE)), W / 2, 960, sc, -3)

def scene3(c, t):
    place(c, LEAF, 780, 820, 1.25, 8, 0.16)  # subliminal, out of focus
    # drop splash -> product
    if t < 0.6:
        place(c, DROP, W / 2, 900, 1 + t * 2, 0, 1 - t / 0.6)
    sp = back(prog(t, 0.25, 0.45), 1.6)
    place(c, PRODUCT, W / 2, 900, 0.8 * sp, -3 + 3 * sp)
    # title block
    tb = back(prog(t, 0.45, 0.3))
    place(c, cached("hampa_t", lambda: text_img("HAMPA", 120, WHITE, F_BLACK, bg=GREEN_D, seed=41)), W / 2, 200, tb, -2)
    place(c, cached("hampa_s", lambda: text_img("Joint & Knee Pain Relief Oil", 54, INK, F_BOLD, bg=PAPER, seed=42, tape=True)), W / 2, 330, back(prog(t, 0.8, 0.3)), 1.5)
    # ingredients orbit
    ing = [(3.75, "Vijaya Leaf", 285, 1470, GREEN_D), (4.81, "Hemp Seed Oil", 790, 1470, AMBER_D),
           (6.22, "Ashwagandha", 285, 1600, INK), (6.95, "Arnica", 790, 1600, GREEN)]
    for i, (it, name, x, y, col) in enumerate(ing):
        if t > it:
            s = back(prog(t, it, 0.3), 2.8)
            bob = math.sin(t * 2.2 + i) * 10
            place(c, cached("ing%d" % i, lambda name=name, col=col, i=i: text_img(name, 46, WHITE, F_BLACK, bg=col, seed=50 + i)), x, y + bob, s, (-4, 3, 2, -3)[i])
    # non-sticky stamp + tick
    if t > 8.3:
        s = back(prog(t, 8.3, 0.25), 3)
        place(c, cached("ns", lambda: text_img("NON-STICKY", 100, WHITE, F_BLACK, bg=AMBER, seed=61)), W / 2 - 40, 1770, s, -4)
        tick(c, 860, 1790, 110, prog(t, 8.45, 0.4))

def scene4(c, t):
    # crowd returns and converges to the drop
    conv = ease_in(prog(t, 1.2, 0.9))
    if t < 2.2:
        for i, (x, y, k) in enumerate(CROWD):
            sc = back(prog(t, i * 0.012, 0.25))
            nx = x + (W / 2 - x) * conv
            ny = y - 40 * math.sin(t * 5 + i) * (1 - conv) + (900 - y) * conv
            place(c, PEOPLE[k], nx, ny, sc * (1 - 0.8 * conv))
        place(c, cached("s4a", lambda: text_img("20 CRORE aching joints", 66, WHITE, F_BLACK, bg=INK, seed=70)), W / 2, 330, back(prog(t, 0.25, 0.3)), -2)
    if 1.6 < t < 2.5:
        place(c, DROP, W / 2, 900, 0.6 + 1.6 * prog(t, 1.6, 0.8))
    # product slam
    if t > 2.3:
        s = back(prog(t, 2.35, 0.3), 3)
        push = 1 + 0.06 * prog(t, 2.7, 7)
        r = random.Random(9)
        d = ImageDraw.Draw(c)
        k = ease_out(prog(t, 2.4, 1.2))
        for j in range(40):  # confetti
            a = r.uniform(0, 2 * math.pi); dist = 700 * k * r.uniform(0.3, 1)
            x, y = W / 2 + math.cos(a) * dist, 860 + math.sin(a) * dist + 300 * k * k
            col = (AMBER, GREEN, AMBER_L, GREEN_D)[j % 4]
            sz = r.uniform(16, 34)
            d.rectangle([x, y, x + sz, y + sz * 0.6], fill=col + (255,))
        place(c, PRODUCT, W / 2, 870, 0.82 * s * push, 0)
        marker_circle(c, W / 2, 870, 270 * push, 510 * push, prog(t, 2.8, 0.5), 14, INK, seed=4)
    if t > 4.15:
        place(c, cached("s4b", lambda: text_img("1000+ customers trust it", 66, WHITE, F_BLACK, bg=GREEN_D, seed=71)), W / 2, 180, back(prog(t, 4.15, 0.3)), -2)
    # lower third band
    if t > 6.6:
        b = ease_out(prog(t, 6.6, 0.3))
        ImageDraw.Draw(c).rectangle([0, H - 520 * b, W, H], fill=GREEN_D + (255,))
        place(c, cached("s4c", lambda: text_img("Free delivery  ·  Cash on delivery", 50, WHITE, F_BOLD)), W / 2, H - 430, b)
    if t > 7.95:
        s = back(prog(t, 7.95, 0.3), 3) * (1 + 0.04 * math.sin((t - 8) * 9))
        btn = cached("btn", lambda: _button())
        place(c, btn, W / 2, H - 250, s)

def _button():
    f = font(F_BLACK, 104)
    txt = "SHOP NOW"
    bb = f.getbbox(txt)
    w, h = bb[2] - bb[0] + 160, bb[3] - bb[1] + 80
    im = Image.new("RGBA", (w + 20, h + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([10, 14, w + 10, h + 14], radius=h // 2, fill=(0, 0, 0, 90))
    d.rounded_rectangle([4, 4, w + 4, h + 4], radius=h // 2, fill=AMBER_L + (255,))
    d.text((4 + 80 - bb[0], 4 + 40 - bb[1]), txt, font=f, fill=INK + (255,))
    return im

SCENES = [(scene1, BG), (scene2, BG_AMBER), (scene3, BG_GREEN), (scene4, BG)]


def frame(t):
    si = min(3, int(t // 10))
    lt = t - si * 10
    fn, bg = SCENES[si]
    c = bg.copy()
    fn(c, lt)
    arr = np.asarray(c.convert("RGB")).astype(np.float32)
    # shake on big impacts
    for imp in (4.55, 17.62, 32.35):
        if 0 <= t - imp < 0.25:
            k = (1 - (t - imp) / 0.25) * 18
            arr = np.roll(arr, (int(random.uniform(-k, k)), int(random.uniform(-k, k))), (0, 1))
    # whip-pan motion blur at scene boundaries
    for b in (10, 20, 30):
        d = t - b
        if -0.22 < d < 0.22:
            k = 1 - abs(d) / 0.22
            shift = int(k * 240) * (-1 if d < 0 else 1)
            acc = np.zeros_like(arr)
            n = 8
            for j in range(n):
                acc += np.roll(arr, int(shift * j / n) + (int(-W * k * 0.5) if d < 0 else int(W * k * 0.5)), axis=1)
            arr = acc / n
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------- SFX bed ----------
def sfx():
    n = int(DUR * SR)
    out = np.zeros(n, np.float32)
    t = np.arange(n) / SR

    def add(at, sig, gain=1.0):
        i = int(at * SR)
        j = min(n, i + len(sig))
        out[i:j] += sig[: j - i] * gain

    def env(d, a=0.005, r=None):
        m = int(d * SR); x = np.arange(m) / SR
        e = np.minimum(1, x / a) * np.exp(-x / (r or d / 4))
        return x, e

    def boom(d=0.9, f0=70):
        x, e = env(d, 0.002, 0.22)
        f = f0 * np.exp(-x * 3)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * e + rng.normal(0, 0.3, len(x)) * np.exp(-x / 0.02)

    def whoosh(d=0.45, up=True):
        m = int(d * SR); x = np.arange(m) / SR
        noise = rng.normal(0, 1, m)
        k = int(SR * 0.002)
        noise = np.convolve(noise, np.ones(k) / k, "same")
        e = np.sin(np.pi * x / d) ** 2
        return noise * e * 2.2

    def pop(f=900, d=0.08):
        x, e = env(d, 0.001, 0.02)
        return np.sin(2 * np.pi * f * x * (1 - x * 4)) * e

    def tickf():
        x, e = env(0.03, 0.0005, 0.006)
        return rng.normal(0, 1, len(x)) * e

    # music bed: 120 bpm kick + sub pulse
    beat = 0.5
    for k in range(int(DUR / beat)):
        at = k * beat
        if 37.5 < at:
            break
        add(at, boom(0.35, 55), 0.28 if k % 2 == 0 else 0.16)
        if k % 2 == 1:
            add(at, tickf(), 0.12)
    sub = 0.05 * np.sin(2 * np.pi * 41 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * t / 8))
    out += sub.astype(np.float32)

    for at in (0.25, 4.55, 17.6, 32.35):
        add(at, boom(1.0, 75), 0.9)
    for at in (9.78, 19.78, 29.78):
        add(at, whoosh(0.44), 0.35)
    for at in (1.2, 4.9, 10.25, 11.79, 13.83, 15.73, 20.45, 20.8, 23.75, 24.81, 26.22, 26.95, 28.3, 34.15, 36.6, 37.95):
        add(at, pop(random.choice([700, 900, 1100])), 0.35)
    for i in range(54):
        add(1.0 + i * 0.05, tickf(), 0.08)
    add(17.2, whoosh(0.4), 0.35)
    add(31.2, whoosh(0.9), 0.3)
    # final chime
    x, e = env(1.6, 0.003, 0.5)
    add(38.0, (np.sin(2 * np.pi * 880 * x) + 0.5 * np.sin(2 * np.pi * 1320 * x)) * e, 0.18)
    fade = np.ones(n, np.float32); fade[-int(1.2 * SR):] = np.linspace(1, 0, int(1.2 * SR))
    out *= fade
    out /= max(1e-6, np.abs(out).max()) / 0.8
    pcm = (out * 32767).astype(np.int16)
    with wave.open(os.path.join(OUT, "sfx.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    sfx()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(OUT, "video.mp4")], stdin=subprocess.PIPE)
    total = int(DUR * FPS)
    only = os.environ.get("FRAMES")
    for i in range(total):
        t = i / FPS
        if only and not any(abs(t - float(x)) < 0.5 / FPS for x in only.split(",")):
            continue
        f = frame(t)
        if only:
            Image.fromarray(f).resize((W // 3, H // 3)).save(os.path.join(OUT, f"f_{t:05.2f}.png"))
        else:
            p.stdin.write(f.tobytes())
        if i % 150 == 0:
            print("frame", i, "/", total, flush=True)
    p.stdin.close(); p.wait()


if __name__ == "__main__":
    main()
