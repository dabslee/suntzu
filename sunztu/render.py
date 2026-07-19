"""Pygame rendering for DnDCombatEnv.

All pygame state (window, fonts, token cache) lives here, so the environment
itself never imports pygame — headless training has no rendering dependency.
The env constructs a PygameRenderer only for 'human'/'rgb_array' modes.
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pygame

from sunztu.creature import Creature


class PygameRenderer:
    CELL_SIZE = 64
    STATS_AREA_HEIGHT = 220  # Two stat lines plus the action log
    TEXT_COLOR = (200, 200, 200)   # Light grey for text
    AGENT_COLOR = (70, 190, 90)    # Green (Agent 1 seat)
    ENEMY_COLOR = (220, 80, 80)    # Red (Agent 2 seat)
    GRID_COLOR = (50, 50, 50)
    BACKGROUND_COLOR = (0, 0, 0)

    # Condition badges drawn on tokens: (condition, letter, color)
    CONDITION_BADGES = [("prone", "P", (230, 200, 60)),
                        ("hidden", "H", (110, 160, 255)),
                        ("grappled", "G", (240, 140, 40))]

    def __init__(self, env, mode: str):
        """`env` is the DnDCombatEnv being drawn; `mode` is 'human' or 'rgb_array'."""
        self.env = env
        self.mode = mode
        self._token_cache: Dict[Tuple[Optional[str], int], Optional[pygame.Surface]] = {}

        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)  # Grid coordinates / smaller text
        self.label_font = pygame.font.Font(None, 16)  # Token name labels
        self.log_font = pygame.font.Font(None, 18)    # Action log lines

        screen_width = env.map_width * self.CELL_SIZE
        screen_height = env.map_height * self.CELL_SIZE + self.STATS_AREA_HEIGHT

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.render_surface: Optional[pygame.Surface] = None
        if mode == 'human':
            pygame.init()
            pygame.display.set_caption("D&D Combat Environment")
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
        else:  # rgb_array
            self.render_surface = pygame.Surface((screen_width, screen_height))

    # --- Public API -------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        """Draw one frame. Returns an (H, W, 3) array in rgb_array mode."""
        if self.mode == 'human':
            if self.screen is None:  # window was closed
                return None
            self._draw_frame(self.screen)
            self._export_frame_if_configured()
            pygame.display.flip()
            self.clock.tick(self.env.metadata.get('render_fps', 4))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            return None

        self._draw_frame(self.render_surface)
        # Convert surface to numpy array and transpose to (H, W, C)
        return np.transpose(pygame.surfarray.array3d(self.render_surface), axes=(1, 0, 2))

    def close(self) -> None:
        if self.mode == 'human' and self.screen is not None:
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception as e:
                print(f"Error during pygame shutdown: {e}")
            self.screen = None  # Mark as closed

    # --- Internals --------------------------------------------------------

    def _export_frame_if_configured(self) -> None:
        if not self.env.export_frames_path:
            return
        filename = os.path.join(
            self.env.export_frames_path,
            f"e{self.env.current_episode_num:03d}_s{self.env.current_episode_steps:04d}.png")
        try:
            pygame.image.save(self.screen, filename)
        except Exception as e:
            print(f"Error saving frame {filename}: {e}")

    def _get_token_surface(self, creature: Creature, size: int) -> Optional[pygame.Surface]:
        """Load and cache the creature's token image scaled to `size`."""
        key = (creature.token_image, size)
        if key in self._token_cache:
            return self._token_cache[key]
        surf = None
        if creature.token_image and os.path.exists(creature.token_image):
            try:
                img = pygame.image.load(creature.token_image)
                try:
                    img = img.convert_alpha()
                except pygame.error:
                    pass  # No display initialized (rgb_array mode) — raw surface is fine
                surf = pygame.transform.smoothscale(img, (size, size))
            except Exception as e:
                print(f"Warning: could not load token image {creature.token_image}: {e}")
        self._token_cache[key] = surf
        return surf

    def _draw_creature_token(self, surface: pygame.Surface, creature: Creature,
                             border_color: Tuple[int, int, int]) -> None:
        cs = self.CELL_SIZE
        cell = pygame.Rect(creature.position[0] * cs, creature.position[1] * cs, cs, cs)
        label_h = 15
        token_size = cs - label_h - 6

        token = self._get_token_surface(creature, token_size)
        if token is not None:
            if creature.has_condition("hidden"):
                token = token.copy()
                token.set_alpha(110)  # ghosted while hidden
            surface.blit(token, (cell.x + (cs - token_size) // 2, cell.y + 3))
        else:
            # Fallback: colored square if the token image is missing
            pygame.draw.rect(surface, border_color,
                             pygame.Rect(cell.x + 5, cell.y + 5, cs - 10, cs - 10 - label_h))

        # Role border ring (green = Agent 1, red = Agent 2)
        pygame.draw.rect(surface, border_color, cell, 2)

        # Small name label underneath the token
        label = self.label_font.render(creature.display_name, True, (235, 235, 235))
        label_bg = pygame.Rect(cell.x + 1, cell.bottom - label_h - 1, cs - 2, label_h)
        pygame.draw.rect(surface, (15, 15, 15), label_bg)
        surface.blit(label, (cell.centerx - label.get_width() // 2,
                             label_bg.y + (label_h - label.get_height()) // 2 + 1))

        # Condition badges along the top edge of the cell
        badges = [(ch, col) for cond, ch, col in self.CONDITION_BADGES
                  if creature.has_condition(cond)]
        if creature.is_dodging:
            badges.append(("D", (90, 220, 220)))
        bx = cell.x + 9
        for ch, col in badges:
            pygame.draw.circle(surface, col, (bx, cell.y + 9), 7)
            ch_surf = self.label_font.render(ch, True, (0, 0, 0))
            surface.blit(ch_surf, (bx - ch_surf.get_width() // 2,
                                   cell.y + 9 - ch_surf.get_height() // 2))
            bx += 16

    def _draw_frame(self, surface: pygame.Surface) -> None:
        env = self.env
        surface.fill(self.BACKGROUND_COLOR)

        # Grid
        for x in range(env.map_width):
            for y in range(env.map_height):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE,
                                   self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(surface, self.GRID_COLOR, rect, 1)

        # Creature tokens
        if env.enemy.is_alive:
            self._draw_creature_token(surface, env.enemy, self.ENEMY_COLOR)
        if env.agent.is_alive:
            self._draw_creature_token(surface, env.agent, self.AGENT_COLOR)

        # Stats + action-log panel
        panel_y = env.map_height * self.CELL_SIZE
        panel = pygame.Rect(0, panel_y, env.map_width * self.CELL_SIZE, self.STATS_AREA_HEIGHT)
        pygame.draw.rect(surface, (25, 25, 25), panel)

        def stat_line(creature: Creature, role_label: str) -> str:
            conditions = ",".join(sorted(creature.conditions)) if creature.conditions else "-"
            extras = []
            if creature.is_dodging:
                extras.append("dodging")
            if creature.attacks_remaining > 0:
                extras.append(f"{creature.attacks_remaining} atk left")
            extra_str = f"  [{', '.join(extras)}]" if extras else ""
            return (f"{role_label}: HP {creature.current_hp}/{creature.max_hp}  "
                    f"AC {creature.ac}  Spd {creature.speed_remaining}/{creature.speed_total}  "
                    f"Cond: {conditions}{extra_str}")

        surface.blit(self.font.render(stat_line(env.agent, env._role_label('agent')),
                                      True, self.AGENT_COLOR), (10, panel_y + 8))
        surface.blit(self.font.render(stat_line(env.enemy, env._role_label('enemy')),
                                      True, self.ENEMY_COLOR), (10, panel_y + 32))
        pygame.draw.line(surface, (60, 60, 60),
                         (10, panel_y + 56), (panel.right - 10, panel_y + 56))

        # Action log — most recent events at the bottom
        log_top = panel_y + 62
        line_height = 18
        max_lines = (self.STATS_AREA_HEIGHT - 62 - 6) // line_height
        for i, entry in enumerate(list(env.event_log)[-max_lines:]):
            text = self.log_font.render(entry[:92], True, self.TEXT_COLOR)
            surface.blit(text, (10, log_top + i * line_height))
