"""Interactive pygame loop: keyboard -> action -> env.step -> renderer.draw.

The app never touches physics. It only decides *which action* to feed the
environment each frame (from the keyboard, or from an optional `policy`
callable) and hands the resulting Snapshot to the renderer.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import pygame

from lander import LEFT, MAIN, NOOP, RIGHT, LanderEnv
from .renderer import Renderer

Policy = Callable[[Sequence[float]], int]


class App:
    def __init__(self, env: LanderEnv, policy: Optional[Policy] = None,
                 width: int = 960, height: int = 640,
                 auto_reset_delay: float = 0.0) -> None:
        """
        policy           optional callable obs -> action. When given, the
                         keyboard only handles reset/pause/quit; useful for
                         watching a controller fly. None = human keyboard.
        auto_reset_delay seconds to wait after an episode ends before
                         resetting automatically (0 = wait for R).
        """
        self.env = env
        self.policy = policy
        self.width, self.height = width, height
        self.auto_reset_delay = auto_reset_delay
        self.paused = False

    # ------------------------------------------------------------------
    @staticmethod
    def keyboard_action(keys) -> int:
        if keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]:
            return MAIN
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return LEFT
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return RIGHT
        return NOOP

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Lander")
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        renderer = Renderer(self.width, self.height)

        obs = self.env.reset()
        done_since: Optional[float] = None
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        obs = self.env.reset()
                        done_since = None
                        self.paused = False
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused

            if not self.env.done and not self.paused:
                if self.policy is not None:
                    action = int(self.policy(obs))
                else:
                    action = self.keyboard_action(pygame.key.get_pressed())
                obs, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    done_since = pygame.time.get_ticks() / 1000.0

            if (self.env.done and self.auto_reset_delay > 0 and done_since is not None
                    and pygame.time.get_ticks() / 1000.0 - done_since >= self.auto_reset_delay):
                obs = self.env.reset()
                done_since = None

            renderer.draw(screen, self.env.snapshot(), dt=1.0 / self.env.cfg.fps)
            if self.paused:
                self._draw_paused(screen, renderer)
            pygame.display.flip()
            clock.tick(self.env.cfg.fps)

        pygame.quit()

    @staticmethod
    def _draw_paused(screen: pygame.Surface, renderer: Renderer) -> None:
        label = renderer.font_big.render("PAUSED", True, (58, 42, 58))
        rect = label.get_rect(center=(screen.get_width() // 2, int(screen.get_height() * 0.55)))
        renderer._card(screen, rect.inflate(48, 24))
        screen.blit(label, rect)
