"""Pygame renderer for the lander. Draws a `Snapshot`; owns no game logic.

Look & feel: a warm "paper-cut" daylight world instead of the usual black
void - apricot sky, layered plum hills, a cream pod with a teal band, soft
round flame puffs and rounded HUD cards.
"""
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

import pygame

from lander import LEFT, MAIN, RIGHT, Snapshot
from lander.env import (STATUS_CRASHED, STATUS_FLYING, STATUS_LANDED_OFF,
                        STATUS_LANDED_PAD, STATUS_OUT_OF_BOUNDS,
                        STATUS_TIMEOUT)
from lander.physics import (body_to_world, hull_world, leg_roots_world,
                            leg_tips_world, right_vector, up_vector)

Color = Tuple[int, int, int]

# ---------------------------------------------------------------- palette
SKY_TOP: Color = (247, 226, 199)
SKY_BOTTOM: Color = (244, 190, 170)
SUN: Color = (255, 243, 214)
SUN_HALO: Color = (255, 230, 190)
HILL_FAR: Color = (232, 176, 158)
HILL_MID: Color = (206, 133, 128)
GROUND: Color = (104, 72, 92)
GROUND_EDGE: Color = (146, 102, 122)
PAD: Color = (31, 168, 160)
PAD_DARK: Color = (20, 120, 115)
PAD_STRIPE: Color = (255, 245, 226)
BEACON: Color = (255, 214, 102)
HULL: Color = (255, 245, 230)
HULL_SHADE: Color = (228, 208, 190)
HULL_OUTLINE: Color = (72, 50, 66)
BAND: Color = (31, 168, 160)
WINDOW: Color = (47, 62, 80)
WINDOW_GLINT: Color = (170, 200, 220)
LEG: Color = (72, 50, 66)
FOOT: Color = (255, 214, 102)
FLAME_CORE: Color = (255, 236, 160)
FLAME_MID: Color = (255, 190, 80)
FLAME_OUT: Color = (255, 120, 70)
TRAIL: Color = (255, 255, 255)
TEXT: Color = (58, 42, 58)
TEXT_SOFT: Color = (120, 96, 112)
CARD: Tuple[int, int, int, int] = (255, 250, 242, 210)
ACCENT: Color = (31, 168, 160)
GOOD: Color = (46, 160, 110)
WARN: Color = (232, 120, 60)
BAD: Color = (214, 70, 80)
MOTE: Tuple[int, int, int, int] = (255, 255, 255, 90)


class Renderer:
    def __init__(self, width: int = 960, height: int = 640) -> None:
        self.width = width
        self.height = height
        self._sky_cache: pygame.Surface | None = None
        self._hills_cache: pygame.Surface | None = None
        self._hills_seed = None
        self._flicker = random.Random(7)
        self._t = 0.0
        pygame.font.init()
        self.font = pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", 15)
        self.font_small = pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", 12)
        self.font_big = pygame.font.SysFont("dejavusans,avenir,helvetica,arial", 34, bold=True)
        self.font_title = pygame.font.SysFont("dejavusans,avenir,helvetica,arial", 14, bold=True)
        # decorative dust motes drifting in the sky
        rng = random.Random(3)
        self._motes = [(rng.random(), rng.random(), rng.uniform(1.5, 3.5), rng.uniform(0.2, 0.6))
                       for _ in range(28)]

    # ------------------------------------------------------------ mapping
    def _scale(self, snap: Snapshot) -> float:
        cfg = snap.config
        return min(self.width / cfg.world_width, self.height / cfg.world_height)

    def _to_screen(self, snap: Snapshot, p: Tuple[float, float]) -> Tuple[float, float]:
        s = self._scale(snap)
        return (p[0] * s, self.height - p[1] * s)

    # ------------------------------------------------------------ public
    def draw(self, surface: pygame.Surface, snap: Snapshot, dt: float = 1 / 50) -> None:
        self._t += dt
        self._draw_sky(surface, snap)
        self._draw_hills(surface, snap)
        self._draw_terrain(surface, snap)
        self._draw_pad(surface, snap)
        self._draw_trail(surface, snap)
        self._draw_lander(surface, snap)
        self._draw_hud(surface, snap)
        if snap.done:
            self._draw_banner(surface, snap)

    # ------------------------------------------------------------ layers
    def _draw_sky(self, surface: pygame.Surface, snap: Snapshot) -> None:
        if self._sky_cache is None or self._sky_cache.get_size() != surface.get_size():
            sky = pygame.Surface(surface.get_size())
            h = sky.get_height()
            for y in range(h):
                t = y / max(1, h - 1)
                t = t * t * (3 - 2 * t)  # smoothstep for a softer horizon
                c = tuple(int(SKY_TOP[i] + (SKY_BOTTOM[i] - SKY_TOP[i]) * t) for i in range(3))
                pygame.draw.line(sky, c, (0, y), (sky.get_width(), y))
            # a big soft sun
            cx, cy = int(sky.get_width() * 0.60), int(h * 0.20)
            halo = pygame.Surface((320, 320), pygame.SRCALPHA)
            for r, a in ((160, 18), (130, 26), (100, 40)):
                pygame.draw.circle(halo, (*SUN_HALO, a), (160, 160), r)
            sky.blit(halo, (cx - 160, cy - 160))
            pygame.draw.circle(sky, SUN, (cx, cy), 58)
            self._sky_cache = sky
        surface.blit(self._sky_cache, (0, 0))

        # drifting motes
        w, h = surface.get_size()
        mote_layer = pygame.Surface((w, h), pygame.SRCALPHA)
        for (mx, my, r, speed) in self._motes:
            x = ((mx + self._t * 0.01 * speed) % 1.0) * w
            y = (my + 0.02 * math.sin(self._t * speed + mx * 10)) * h * 0.7
            pygame.draw.circle(mote_layer, MOTE, (int(x), int(y)), int(r))
        surface.blit(mote_layer, (0, 0))

    def _draw_hills(self, surface: pygame.Surface, snap: Snapshot) -> None:
        """Two decorative parallax hill bands behind the real terrain."""
        key = (id(snap.terrain), surface.get_size())
        if self._hills_cache is None or self._hills_seed != key:
            w, h = surface.get_size()
            layer = pygame.Surface((w, h), pygame.SRCALPHA)
            rng = random.Random(sum(int(v * 100) for v in snap.terrain.ys))
            for color, base, amp, freq in ((HILL_FAR, 0.62, 0.05, 1.7), (HILL_MID, 0.70, 0.04, 2.6)):
                phase = rng.uniform(0, 6.28)
                pts = []
                for i in range(0, w + 16, 16):
                    t = i / w
                    y = h * (base - amp * (math.sin(t * freq * 6.28 + phase)
                                          + 0.5 * math.sin(t * freq * 13.1 + phase * 2)))
                    pts.append((i, y))
                pts += [(w, h), (0, h)]
                pygame.draw.polygon(layer, color, pts)
            self._hills_cache = layer
            self._hills_seed = key
        surface.blit(self._hills_cache, (0, 0))

    def _draw_terrain(self, surface: pygame.Surface, snap: Snapshot) -> None:
        t = snap.terrain
        pts = [self._to_screen(snap, (x, y)) for x, y in zip(t.xs, t.ys)]
        poly = pts + [(self.width, self.height), (0, self.height)]
        pygame.draw.polygon(surface, GROUND, poly)
        pygame.draw.lines(surface, GROUND_EDGE, False, pts, 4)

    def _draw_pad(self, surface: pygame.Surface, snap: Snapshot) -> None:
        t = snap.terrain
        s = self._scale(snap)
        x0, y = self._to_screen(snap, (t.pad_x0, t.pad_y))
        x1, _ = self._to_screen(snap, (t.pad_x1, t.pad_y))
        thick = max(8, int(0.22 * s))
        rect = pygame.Rect(int(x0), int(y) - thick + 2, int(x1 - x0), thick)
        pygame.draw.rect(surface, PAD_DARK, rect.move(0, 3), border_radius=6)
        pygame.draw.rect(surface, PAD, rect, border_radius=6)
        # cream stripes
        n = 6
        seg = rect.width / n
        for i in range(n):
            if i % 2 == 0:
                sr = pygame.Rect(int(rect.x + i * seg + 4), rect.y + 3, int(seg - 8), thick - 6)
                pygame.draw.rect(surface, PAD_STRIPE, sr, border_radius=3)
        # beacons pulsing at each end
        pulse = 0.5 + 0.5 * math.sin(self._t * 5.0)
        for bx in (rect.left + 6, rect.right - 6):
            glow = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*BEACON, int(40 + 90 * pulse)), (20, 20), 14)
            surface.blit(glow, (bx - 20, rect.y - 24))
            pygame.draw.line(surface, HULL_OUTLINE, (bx, rect.y), (bx, rect.y - 10), 2)
            pygame.draw.circle(surface, BEACON, (bx, rect.y - 12), 4)

    def _draw_trail(self, surface: pygame.Surface, snap: Snapshot) -> None:
        pts = snap.trail
        if len(pts) < 2:
            return
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        n = len(pts)
        for i in range(0, n, 3):
            a = int(120 * i / n)
            px, py = self._to_screen(snap, pts[i])
            pygame.draw.circle(layer, (*TRAIL, a), (int(px), int(py)), 2)
        surface.blit(layer, (0, 0))

    def _draw_flame(self, surface: pygame.Surface, origin: Tuple[float, float],
                    direction: Tuple[float, float], length_px: float, width_px: float) -> None:
        """Soft stacked puffs from `origin` along `direction` (screen coords)."""
        flick = self._flicker
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        n = 5
        for i in range(n):
            t = (i + 0.5) / n
            jitter = flick.uniform(0.85, 1.15)
            cx = origin[0] + direction[0] * length_px * t * jitter
            cy = origin[1] + direction[1] * length_px * t * jitter
            r = width_px * (1.0 - 0.75 * t) * flick.uniform(0.9, 1.1)
            color = FLAME_CORE if t < 0.3 else FLAME_MID if t < 0.65 else FLAME_OUT
            pygame.draw.circle(layer, (*color, int(230 * (1 - t * 0.6))), (int(cx), int(cy)), int(r))
        surface.blit(layer, (0, 0))

    def _body_point(self, snap: Snapshot, p: Tuple[float, float]) -> Tuple[float, float]:
        """Screen position of a point given in the lander's body frame."""
        return self._to_screen(snap, body_to_world(snap.state, p))

    def _draw_lander(self, surface: pygame.Surface, snap: Snapshot) -> None:
        st, cfg, terrain = snap.state, snap.config, snap.terrain
        s = self._scale(snap)
        crashed = snap.status in (STATUS_CRASHED, STATUS_OUT_OF_BOUNDS)

        # ---- flames (behind the body) --------------------------------
        if snap.status == STATUS_FLYING:
            ux, uy = up_vector(st)
            rx, ry = right_vector(st)
            if snap.action == MAIN:
                origin = self._body_point(snap, (0.0, -0.32))
                # screen y is flipped, hence (-ux, +uy) for "down the body axis"
                self._draw_flame(surface, origin, (-ux, uy), 1.3 * s, 0.32 * s)
            elif snap.action in (LEFT, RIGHT):
                # LEFT pushes the body toward -x, so its exhaust leaves the
                # right flank toward +x (and vice-versa).
                side = 1.0 if snap.action == LEFT else -1.0
                origin = self._body_point(snap, (side * cfg.side_engine_x, cfg.side_engine_y))
                self._draw_flame(surface, origin, (side * rx, -side * ry), 0.7 * s, 0.16 * s)

        # ---- legs ------------------------------------------------------
        roots = leg_roots_world(st, cfg)
        tips = leg_tips_world(st, cfg)
        foot_r = max(3, int(0.09 * s))
        for (rxw, ryw), (tx, ty) in zip(roots, tips):
            # the visual foot never sinks below ground: shows leg compression
            ground = terrain.height_at(tx)
            ty_vis = max(ty, ground) if not crashed else ty
            a = self._to_screen(snap, (rxw, ryw))
            b = self._to_screen(snap, (tx, ty_vis))
            pygame.draw.line(surface, LEG, a, b, foot_r)
            pygame.draw.circle(surface, FOOT, (int(b[0]), int(b[1])), foot_r)
            pygame.draw.circle(surface, LEG, (int(b[0]), int(b[1])), foot_r, 2)

        # ---- hull --------------------------------------------------------
        hull = [self._to_screen(snap, p) for p in hull_world(st, cfg)]
        body_color = HULL if not crashed else (200, 150, 150)
        shade_color = HULL_SHADE if not crashed else (180, 130, 130)
        pygame.draw.polygon(surface, body_color, hull)
        # shaded right half for a bit of volume
        shade = [self._body_point(snap, p) for p in ((0.0, 0.62), (0.42, 0.28), (0.45, -0.32), (0.0, -0.32))]
        pygame.draw.polygon(surface, shade_color, shade)
        pygame.draw.polygon(surface, HULL_OUTLINE, hull, max(2, int(0.05 * s)))
        # teal band across the middle
        band = [self._body_point(snap, p) for p in ((-0.44, -0.05), (0.44, -0.05), (0.45, -0.20), (-0.45, -0.20))]
        pygame.draw.polygon(surface, BAND, band)
        # porthole with a glint
        wc = self._body_point(snap, (0.0, 0.18))
        pygame.draw.circle(surface, WINDOW, (int(wc[0]), int(wc[1])), int(0.15 * s))
        pygame.draw.circle(surface, HULL_OUTLINE, (int(wc[0]), int(wc[1])), int(0.15 * s), 2)
        gc = self._body_point(snap, (-0.05, 0.23))
        pygame.draw.circle(surface, WINDOW_GLINT, (int(gc[0]), int(gc[1])), max(2, int(0.045 * s)))

        # ---- settle ring while both legs are down ------------------------
        if snap.settle_progress > 0 and snap.status == STATUS_FLYING:
            c = self._to_screen(snap, (st.x, st.y))
            r = int(1.1 * s)
            rect = pygame.Rect(int(c[0]) - r, int(c[1]) - r, 2 * r, 2 * r)
            pygame.draw.arc(surface, ACCENT, rect, math.pi / 2,
                            math.pi / 2 + 2 * math.pi * snap.settle_progress, 3)

    # ------------------------------------------------------------ HUD
    def _card(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        card = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(card, CARD, card.get_rect(), border_radius=14)
        surface.blit(card, rect.topleft)

    def _draw_hud(self, surface: pygame.Surface, snap: Snapshot) -> None:
        st = snap.state
        cfg = snap.config
        leg_h = -cfg.leg_tips[0][1]
        altitude = st.y - leg_h - snap.terrain.height_at(st.x)

        # ---- left card: telemetry ------------------------------------------
        rect = pygame.Rect(16, 16, 236, 176)
        self._card(surface, rect)
        title = self.font_title.render("TELEMETRY", True, ACCENT)
        surface.blit(title, (rect.x + 16, rect.y + 12))

        def color_for(value: float, good: float, bad: float) -> Color:
            v = abs(value)
            return GOOD if v < good else WARN if v < bad else BAD

        rows = [
            ("ALT", f"{altitude:6.2f}", TEXT),
            ("V-X", f"{st.vx:+6.2f}", color_for(st.vx, 0.6, 1.5)),
            ("V-Y", f"{st.vy:+6.2f}", color_for(st.vy, 1.0, cfg.max_landing_speed)),
            ("ANG", f"{math.degrees(st.angle):+6.1f}°", color_for(st.angle, 0.15, 0.4)),
            ("Ω  ", f"{st.angular_velocity:+6.2f}", color_for(st.angular_velocity, 0.3, 1.0)),
        ]
        y = rect.y + 40
        for label, value, color in rows:
            surface.blit(self.font.render(label, True, TEXT_SOFT), (rect.x + 16, y))
            surface.blit(self.font.render(value, True, color), (rect.x + 96, y))
            y += 22
        legs = f"legs  {'L' if st.left_contact else '·'} {'R' if st.right_contact else '·'}"
        surface.blit(self.font_small.render(legs, True, TEXT_SOFT), (rect.x + 16, y + 4))

        # ---- right card: score + engines + attitude ------------------------
        rect2 = pygame.Rect(self.width - 16 - 236, 16, 236, 176)
        self._card(surface, rect2)
        surface.blit(self.font_title.render("MISSION", True, ACCENT), (rect2.x + 16, rect2.y + 12))
        score_col = GOOD if snap.episode_reward >= 0 else BAD
        surface.blit(self.font.render("SCORE", True, TEXT_SOFT), (rect2.x + 16, rect2.y + 40))
        surface.blit(self.font.render(f"{snap.episode_reward:+8.1f}", True, score_col), (rect2.x + 80, rect2.y + 40))
        surface.blit(self.font.render("STEP", True, TEXT_SOFT), (rect2.x + 16, rect2.y + 62))
        surface.blit(self.font.render(f"{snap.step_index:5d}/{cfg.max_steps}", True, TEXT), (rect2.x + 80, rect2.y + 62))
        surface.blit(self.font.render("FUEL", True, TEXT_SOFT), (rect2.x + 16, rect2.y + 84))
        surface.blit(self.font.render(f"{snap.fuel_used:8.1f}", True, TEXT), (rect2.x + 80, rect2.y + 84))

        # engine lamps
        lamps = (("◀", LEFT), ("▲", MAIN), ("▶", RIGHT))
        lx = rect2.x + 20
        ly = rect2.y + 126
        for i, (glyph, act) in enumerate(lamps):
            on = snap.action == act and not snap.done
            cx = lx + i * 40 + 14
            pygame.draw.circle(surface, FLAME_MID if on else (225, 212, 200), (cx, ly + 12), 14)
            pygame.draw.circle(surface, HULL_OUTLINE, (cx, ly + 12), 14, 2)
            g = self.font.render(glyph, True, HULL_OUTLINE)
            surface.blit(g, g.get_rect(center=(cx, ly + 12)))

        # attitude dial
        cx, cy, r = rect2.right - 44, rect2.y + 138, 26
        pygame.draw.circle(surface, (235, 226, 214), (cx, cy), r)
        pygame.draw.circle(surface, HULL_OUTLINE, (cx, cy), r, 2)
        pygame.draw.line(surface, TEXT_SOFT, (cx - r + 4, cy), (cx + r - 4, cy), 1)
        ax = cx - math.sin(st.angle) * (r - 6)
        ay = cy - math.cos(st.angle) * (r - 6)
        pygame.draw.line(surface, ACCENT if abs(st.angle) < 0.3 else BAD, (cx, cy), (ax, ay), 3)
        pygame.draw.circle(surface, HULL_OUTLINE, (cx, cy), 3)

        # ---- controls hint -------------------------------------------------
        hint = "↑ / W / SPACE  main engine    ← / A  left engine    → / D  right engine    R  reset    P  pause    ESC  quit"
        h = self.font_small.render(hint, True, TEXT)
        hr = h.get_rect(midbottom=(self.width // 2, self.height - 10))
        bg = pygame.Rect(hr.x - 12, hr.y - 6, hr.width + 24, hr.height + 12)
        self._card(surface, bg)
        surface.blit(h, hr)

    def _draw_banner(self, surface: pygame.Surface, snap: Snapshot) -> None:
        if snap.status == STATUS_LANDED_PAD:
            text, sub, col = "TOUCHDOWN", "perfect - right on the pad", GOOD
        elif snap.status == STATUS_LANDED_OFF:
            text, sub, col = "DOWN SAFE", "…but you missed the pad", WARN
        elif snap.status == STATUS_CRASHED:
            text, sub, col = "CRASHED", snap.crash_reason, BAD
        elif snap.status == STATUS_OUT_OF_BOUNDS:
            text, sub, col = "LOST", snap.crash_reason, BAD
        elif snap.status == STATUS_TIMEOUT:
            text, sub, col = "OUT OF TIME", "the episode hit its step limit", WARN
        else:
            return
        rect = pygame.Rect(0, 0, 420, 132)
        rect.center = (self.width // 2, int(self.height * 0.38))
        self._card(surface, rect)
        pygame.draw.rect(surface, col, pygame.Rect(rect.x, rect.y, 8, rect.height), border_radius=14)
        t1 = self.font_big.render(text, True, col)
        surface.blit(t1, t1.get_rect(midtop=(rect.centerx + 4, rect.y + 16)))
        t2 = self.font.render(sub, True, TEXT)
        surface.blit(t2, t2.get_rect(midtop=(rect.centerx + 4, rect.y + 62)))
        t3 = self.font_small.render(f"episode score {snap.episode_reward:+.1f}   ·   press R to fly again",
                                    True, TEXT_SOFT)
        surface.blit(t3, t3.get_rect(midtop=(rect.centerx + 4, rect.y + 94)))
