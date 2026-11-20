import math
import time
import sys
import shutil
import numpy as np

RAMP = np.array(list(" .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"))

def build_stars(count=85, seed=271):
    rng = np.random.default_rng(seed)
    stars = []
    palette = [(180, 235, 255), (255, 190, 235), (255, 235, 195), (160, 220, 255)]
    for i in range(count):
        stars.append((
            float(rng.uniform(0.02, 0.98)),
            float(rng.uniform(0.02, 0.98)),
            float(rng.uniform(0.0, 2.0 * math.pi)),
            float(rng.uniform(1.4, 3.6)),
            i < 12,
            palette[i % len(palette)]
        ))
    return stars

def main():
    stars = build_stars()
    light_dir = np.array([0.35, -0.52, 0.78], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    half_vec = light_dir + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_vec /= np.linalg.norm(half_vec)
    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    alt_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    N = 2000
    M = 8
    E = math.e
    theta = np.linspace(0.0, 8.0 * math.pi, N, dtype=np.float32)
    taper = np.maximum(0.0, np.sin(np.linspace(0.0, math.pi, N, dtype=np.float32))) ** 0.35
    phi = np.linspace(0.0, 2.0 * math.pi, M, endpoint=False, dtype=np.float32)[None, :]
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    r_major = (0.025 * taper)[:, None]
    r_minor = (0.011 * taper)[:, None]
    s_norm = theta[:, None] / (8.0 * math.pi)
    spark_idx = np.arange(35, N - 35, 23)

    sys.stdout.write("\033[?1049h\033[?25l\033[48;2;0;0;0m\033[2J")
    sys.stdout.flush()
    t0 = time.perf_counter()

    try:
        while True:
            frame_start = time.perf_counter()
            t = frame_start - t0
            fade = min(1.0, t / 0.30)

            cols, rows = shutil.get_terminal_size((136, 56))
            W, H = max(56, cols), max(24, rows - 1)

            z_buf = np.full((H, W), np.inf, dtype=np.float32)
            char_buf = np.full((H, W), ' ', dtype='<U1')
            col_buf = np.zeros((H, W, 3), dtype=np.uint8)

            aspect = 2.05
            scale = min(W / aspect, H) * 0.47
            cx, cy = W * 0.5, H * 0.5

            w2 = E + 0.08 * math.sin(t * 0.45)
            w3 = (E * E) - 0.12 * math.cos(t * 0.35)

            p1 = t * 0.85
            p2 = -t * 0.65
            p3 = t * 0.95

            x0 = np.cos(theta) + 0.54 * np.cos(w2 * theta + p1) + 0.22 * np.cos(w3 * theta + p2)
            y0 = np.sin(theta) + 0.54 * np.sin(w2 * theta + p1) + 0.22 * np.sin(w3 * theta + p2)
            z0 = (
                0.82 * np.sin((E - 1.0) * theta + p1 * 0.75)
                + 0.48 * np.cos((E + 1.0) * theta + p3)
                + 0.22 * np.sin(2.0 * theta - p2)
            )

            pts = np.stack([x0, y0, z0], axis=-1) * 0.52

            ay = t * 0.42
            ax = 0.46 + 0.20 * math.sin(t * 0.32)
            az = t * 0.18

            cy_a, sy_a = math.cos(ay), math.sin(ay)
            cx_a, sx_a = math.cos(ax), math.sin(ax)
            cz_a, sz_a = math.cos(az), math.sin(az)

            px = pts[:, 0] * cy_a + pts[:, 2] * sy_a
            pz = -pts[:, 0] * sy_a + pts[:, 2] * cy_a
            py = pts[:, 1]
            py2 = py * cx_a - pz * sx_a
            pz2 = py * sx_a + pz * cx_a
            px3 = px * cz_a - py2 * sz_a
            py3 = px * sz_a + py2 * cz_a
            pz3 = pz2

            curve = np.stack([px3, py3, pz3], axis=-1)

            tang = np.gradient(curve, axis=0)
            tang /= (np.linalg.norm(tang, axis=-1, keepdims=True) + 1e-8)
            norm_v = np.cross(tang, up_ref)
            n_len = np.linalg.norm(norm_v, axis=-1, keepdims=True)
            norm_v = np.where(n_len < 1e-3, np.cross(tang, alt_ref), norm_v)
            norm_v /= (np.linalg.norm(norm_v, axis=-1, keepdims=True) + 1e-8)
            binorm_v = np.cross(tang, norm_v)

            surf_normals = norm_v[:, None, :] * cos_p[..., None] + binorm_v[:, None, :] * sin_p[..., None]
            surf_normals /= (np.linalg.norm(surf_normals, axis=-1, keepdims=True) + 1e-8)
            surf_pts = (
                curve[:, None, :]
                + norm_v[:, None, :] * (r_major * cos_p)[..., None]
                + binorm_v[:, None, :] * (r_minor * sin_p)[..., None]
            )

            ndotl = np.maximum(0.0, np.sum(surf_normals * light_dir, axis=-1))
            ndoth = np.maximum(0.0, np.sum(surf_normals * half_vec, axis=-1))
            spec = (ndoth ** 18) * 0.85
            fresnel = (1.0 - np.abs(surf_normals[..., 2])) ** 1.45

            # Traveling soliton light packets racing along the Euler curve
            pulse_wave = (
                np.exp(-((np.sin((s_norm * 3.0 - t * 0.55) * math.pi)) ** 2) * 18.0)
                + 0.75 * np.exp(-((np.sin((s_norm * 5.0 + t * 0.38) * math.pi)) ** 2) * 24.0)
            )

            depth_fade = np.clip(1.02 - 0.34 * surf_pts[..., 2], 0.36, 1.25)
            lum = np.clip((0.30 + 0.55 * ndotl + 0.36 * fresnel + spec + 0.45 * pulse_wave) * depth_fade * fade, 0.12, 1.0)

            mix = 0.5 + 0.5 * np.sin(theta[:, None] * 0.65 - t * 1.1)
            r_col = (0.04 * (1.0 - mix) + 1.00 * mix) * (0.42 + 0.68 * lum) + 0.55 * pulse_wave + 0.75 * spec
            g_col = (0.94 * (1.0 - mix) + 0.14 * mix) * (0.42 + 0.68 * lum) + 0.48 * pulse_wave + 0.75 * spec
            b_col = (1.00 * (1.0 - mix) + 0.58 * mix) * (0.42 + 0.68 * lum) + 0.32 * pulse_wave + 0.85 * spec

            persp = 4.2 / (4.2 + surf_pts[..., 2])
            sx = np.rint(cx + surf_pts[..., 0] * scale * aspect * persp).astype(np.int32)
            sy = np.rint(cy + surf_pts[..., 1] * scale * persp).astype(np.int32)
            sz = surf_pts[..., 2]

            mask = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & (sz > -3.2)
            sx_m, sy_m, sz_m = sx[mask], sy[mask], sz[mask]
            lum_m = lum[mask]
            r_m, g_m, b_m = r_col[mask], g_col[mask], b_col[mask]

            flat = sy_m * W + sx_m
            order = np.argsort(sz_m)
            flat_s, sz_s, lum_s = flat[order], sz_m[order], lum_m[order]

            uniq_flat, first_idx = np.unique(flat_s, return_index=True)
            z_flat = z_buf.ravel()
            z_flat[uniq_flat] = sz_s[first_idx]
            l_win = lum_s[first_idx]
            c_idx = np.clip((l_win * (len(RAMP) - 1)).astype(np.int32), 6, len(RAMP) - 1)
            char_buf.ravel()[uniq_flat] = RAMP[c_idx]

            rgb_win = np.stack([r_m[order][first_idx], g_m[order][first_idx], b_m[order][first_idx]], axis=-1)
            col_buf.reshape(-1, 3)[uniq_flat] = np.clip(rgb_win * 255.0, 0, 255).astype(np.uint8)

            # Luminous sparks shedding off the ribbons
            drift = 0.05 * (0.5 + 0.5 * np.sin(t * 2.1 + spark_idx * 0.37))[:, None]
            sp_pts = curve[spark_idx] + norm_v[spark_idx] * drift + binorm_v[spark_idx] * (drift * 0.7)
            sp_persp = 4.2 / (4.2 + sp_pts[:, 2])
            spx = np.rint(cx + sp_pts[:, 0] * scale * aspect * sp_persp).astype(np.int32)
            spy = np.rint(cy + sp_pts[:, 1] * scale * sp_persp).astype(np.int32)
            spz = sp_pts[:, 2]
            for k in range(len(spark_idx)):
                x_i, y_i, z_i = int(spx[k]), int(spy[k]), float(spz[k])
                if 0 <= x_i < W and 0 <= y_i < H and z_i < z_buf[y_i, x_i]:
                    pulse = max(0.0, 0.45 + 0.55 * math.sin(t * 3.2 + k)) * fade
                    char_buf[y_i, x_i] = '*' if pulse > 0.75 else '+'
                    col_buf[y_i, x_i] = (int(240 * pulse), int(248 * pulse), int(255 * pulse))

            for fx, fy, ph, spd, hero, (sr_c, sg_c, sb_c) in stars:
                bx, by = int(fx * (W - 1)), int(fy * (H - 1))
                if char_buf[by, bx] != ' ':
                    continue
                tw = max(0.0, 0.25 + 0.65 * (0.5 + 0.5 * math.sin(t * spd + ph))) * fade
                char_buf[by, bx] = '+' if (hero and tw > 0.70) else '.'
                col_buf[by, bx] = (int(sr_c * tw), int(sg_c * tw), int(sb_c * tw))

            out = ["\033[H\033[48;2;0;0;0m"]
            last_rgb = None
            for r in range(H):
                row_chars = char_buf[r]
                row_cols = col_buf[r]
                for c in range(W):
                    ch = row_chars[c]
                    if ch == ' ':
                        out.append(' ')
                    else:
                        rgb = (int(row_cols[c, 0]), int(row_cols[c, 1]), int(row_cols[c, 2]))
                        if rgb != last_rgb:
                            out.append(f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m")
                            last_rgb = rgb
                        out.append(ch)
                if r < H - 1:
                    out.append("\n")

            sys.stdout.write("".join(out))
            sys.stdout.flush()

            elapsed = time.perf_counter() - frame_start
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[0m\033[?25h\033[?1049l")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
