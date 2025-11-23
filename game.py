# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import os
import random

import pygame
import torch
import numpy as np

# --- Constants ---
VIEW_CONFIGS = {
    'desktop': {'width': 1024, 'height': 768},
    'mobile':  {'width': 450, 'height': 800},
}
INITIAL_VIEW_MODE = 'desktop'

# Gameplay Constants
PLAYER_SPEED = 350
BOOST_SPEED = 900
BOOST_DURATION = 15
BOOST_COOLDOWN = 120
ANIMATION_SPEED = 10
STEP_INTERVAL_WALK = 12
STEP_INTERVAL_RUN = 8

# Shake settings
SHAKE_DISTANCE_THRESHOLD = 200 # How close AI needs to be to start shaking
MAX_SHAKE_INTENSITY = 8        # Max pixel offset for shake

PLAYER_NAMES = ['QuantifiedQuantum', 'Kalamata', 'EmoAImusic', 'MD', 'Torva', 'Haidar', 'BoboBear', 'Mohamed', 'Alucard', 'Kevin', 'Barry', 'Uniqueux', 'JanHoleman', 'TheJAM', 'megansub', 'Dereck', 'Kyle', 'Tuleku', 'Travis', 'Valor', 'Lukey', 'Mosh', 'Alazr', 'Ahmed']
AI_NAMES = ['HAL9000', 'Skynet', 'Predator', 'DeepBlue', 'AlphaGo', 'Watson', 'Siri', 'nAIma', 'Aldan', 'mAIa', 'nAlma', 'gAIl', 'bAIley', 'dAIsy']

# --- Asset Manager ---
class AssetManager:
    """Handles loading, storing, and volume control for all assets."""
    def __init__(self):
        self.animations: Dict[str, List[pygame.Surface]] = {}
        self.ai_image: Optional[pygame.Surface] = None
        self.title_bg: Optional[pygame.Surface] = None 
        self.tiles: List[pygame.Surface] = []
        self.vignette: Optional[pygame.Surface] = None
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self.music_track = None
        self.global_volume = 0.3 
        
        # Fonts
        self.font_sm = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_md = pygame.font.SysFont("Arial", 26)
        self.font_mono = pygame.font.SysFont("Mono", 24)
        self.font_lg = pygame.font.SysFont("Arial", 50)
        self.font_xl = pygame.font.SysFont("Arial", 300)

    def load_all(self, width, height):
        print("Loading Assets...")
        self._load_sprites()
        self.generate_procedural_assets(width, height)
        self._load_audio()

    def _load_sprites(self):
        def load_strip(filename, frames):
            if not os.path.exists(filename): return []
            try:
                sheet = pygame.image.load(filename).convert_alpha()
                h = sheet.get_height()
                return [pygame.transform.scale(sheet.subsurface(i*h, 0, h, h), (64, 64)) for i in range(frames)]
            except: return []

        self.animations['idle_down'] = load_strip("idle_down.png", 2)
        self.animations['idle_up'] = load_strip("idle_up.png", 2)
        self.animations['idle_left'] = load_strip("idle_left.png", 4)
        self.animations['idle_right'] = load_strip("idle_right.png", 4)
        self.animations['walk_down'] = load_strip("walk_down.png", 4)
        self.animations['walk_up'] = load_strip("walk_up.png", 4)
        self.animations['walk_left'] = load_strip("walk_left.png", 4)
        self.animations['walk_right'] = load_strip("walk_right.png", 4)
        
        if os.path.exists("ai_sprite.png"):
            self.ai_image = pygame.transform.scale(pygame.image.load("ai_sprite.png").convert_alpha(), (80, 80))

        if os.path.exists("title_bg.png"):
            try:
                self.title_bg = pygame.image.load("title_bg.png").convert()
                print("Title background loaded.")
            except Exception as e:
                print(f"Background load error: {e}")

    def generate_procedural_assets(self, width, height):
        if not self.tiles:
            colors = [(20,30,40), (60,30,20), (50,50,50), (10,40,20)]
            for base_col in colors:
                s = pygame.Surface((64, 64))
                s.fill(base_col)
                pygame.draw.rect(s, (min(base_col[0]+20,255), min(base_col[1]+20,255), min(base_col[2]+20,255)), (0,0,64,64), 1)
                self.tiles.append(s)

        self.vignette = pygame.Surface((width, height), pygame.SRCALPHA)
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xv, yv = np.meshgrid(x, y)
        dist = np.sqrt(xv**2 + yv**2)
        alpha = (np.minimum((dist - 0.4).clip(0, 1) * 380, 255)).astype(np.uint8).T
        pygame.surfarray.pixels_alpha(self.vignette)[:] = alpha

    def _load_audio(self):
        if os.path.exists("footstep.mp3"):
            try:
                self.sounds['step'] = pygame.mixer.Sound("footstep.mp3")
                self.set_volume(self.global_volume) 
            except pygame.error as e:
                print(f"Warning: Could not load footstep.mp3: {e}")
            
    def play_music(self, track_name):
        if self.music_track == track_name: return
        if os.path.exists(track_name):
            try:
                pygame.mixer.music.load(track_name)
                pygame.mixer.music.set_volume(self.global_volume)
                pygame.mixer.music.play(-1)
                self.music_track = track_name
            except: pass

    def set_volume(self, volume):
        self.global_volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.global_volume)
        for s in self.sounds.values():
            s.set_volume(self.global_volume)

# --- Entities ---
@dataclass
class Entity:
    position: pygame.Vector2
    size: int
    name: str
    color: Tuple[int, int, int] = (255, 255, 255)

class Player(Entity):
    def __init__(self, x, y):
        super().__init__(pygame.Vector2(x, y), 30, random.choice(PLAYER_NAMES), (0, 255, 255))
        self.velocity = pygame.Vector2(0, 0)
        self.speed = PLAYER_SPEED
        self.boost_timer = 0
        self.boost_cooldown = 0
        self.step_timer = 0
        self.trail = []
        self.facing = 'down'
        self.anim_state = 'idle'
        self.frame_index = 0.0

    def reset(self, x, y):
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(0, 0)
        self.trail = []
        self.boost_timer = 0
        self.boost_cooldown = 0
        self.step_timer = 0
        self.name = random.choice(PLAYER_NAMES)

    def update(self, delta, bounds, assets: AssetManager):
        if self.boost_timer > 0:
            self.boost_timer -= 1
            current_speed = BOOST_SPEED
            self.trail.append(self.position.copy())
            if len(self.trail) > 10: self.trail.pop(0)
        else:
            current_speed = PLAYER_SPEED
            if self.trail: self.trail.pop(0)
        
        if self.boost_cooldown > 0: self.boost_cooldown -= 1

        move_vec = pygame.Vector2(0, 0)
        if self.velocity.length() > 0:
            move_vec = self.velocity.normalize() * current_speed
            
            if abs(self.velocity.x) > abs(self.velocity.y):
                self.facing = 'right' if self.velocity.x > 0 else 'left'
            else:
                self.facing = 'down' if self.velocity.y > 0 else 'up'
            self.anim_state = 'walk'

            if self.step_timer <= 0:
                if 'step' in assets.sounds: assets.sounds['step'].play()
                self.step_timer = STEP_INTERVAL_RUN if self.boost_timer > 0 else STEP_INTERVAL_WALK
            else:
                self.step_timer -= 1
        else:
            self.anim_state = 'idle'
            self.step_timer = 0

        if self.boost_timer > 0 and self.velocity.length() == 0:
             dirs = {'right': (1,0), 'left': (-1,0), 'up': (0,-1), 'down': (0,1)}
             d = dirs.get(self.facing, (0,0))
             move_vec = pygame.Vector2(d[0], d[1]) * current_speed

        # Clamp Player to screen boundary
        self.position += move_vec * delta
        self.position.x = max(self.size, min(self.position.x, bounds[0] - self.size))
        self.position.y = max(self.size, min(self.position.y, bounds[1] - self.size))

    def draw(self, screen, assets: AssetManager, offset: pygame.Vector2):
        draw_pos = self.position + offset
        
        for i, pos in enumerate(self.trail):
            ts = self.size * (i / len(self.trail))
            pygame.draw.circle(screen, (0, 200, 255), (int(pos.x + offset.x), int(pos.y + offset.y)), int(ts))

        key = f"{self.anim_state}_{self.facing}"
        frames = assets.animations.get(key, assets.animations.get('idle_down'))
        
        if frames:
            self.frame_index = (self.frame_index + ANIMATION_SPEED * 0.016) % len(frames)
            img = frames[int(self.frame_index)]
            rect = img.get_rect(center=(draw_pos.x, draw_pos.y))
            screen.blit(img, rect)
        else:
            pygame.draw.circle(screen, self.color, (int(draw_pos.x), int(draw_pos.y)), self.size)

        lbl = assets.font_md.render(self.name, True, (255, 255, 0))
        screen.blit(lbl, (draw_pos.x - lbl.get_width()//2, draw_pos.y - 65))

class AI(Entity):
    def __init__(self, x, y):
        super().__init__(pygame.Vector2(x, y), 40, random.choice(AI_NAMES), (255, 0, 0))
        self.level = 1
        self.knowledge = 0
        self.speed_mult = 1.0
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 2), torch.nn.Tanh(),
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.loss_history = []
        self.learning_rate = 5 

    def reset(self, width, height, level):
        self.level = level
        self.knowledge = 0
        self.speed_mult = 1 + (level * 0.4)
        self.size = 40 + (level * 2)
        self.name = random.choice(AI_NAMES)
        
        corners = [(0,0), (width,0), (0,height), (width,height)]
        self.position = pygame.Vector2(random.choice(corners))

    def update(self, player_pos, delta, width, height, frame):
        dx = player_pos.x - self.position.x
        dy = player_pos.y - self.position.y
        
        features = torch.tensor([[player_pos.x/width, player_pos.y/height, self.position.x/width, self.position.y/height]], dtype=torch.float32)
        labels = torch.tensor([[dx/width, dy/height]], dtype=torch.float32)

        output = self.model(features)
        
        if frame % self.learning_rate == 0:
            loss = self.loss_fn(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
            self.loss_history = self.loss_history[-500:]

        move_x = float(output[0][0])
        move_y = float(output[0][1])
        
        self.position.x += (width/2) * move_x * delta * self.speed_mult
        self.position.y += (height/2) * move_y * delta * self.speed_mult

        # --- FIXED: Clamp AI to screen boundary ---
        self.position.x = max(self.size, min(self.position.x, width - self.size))
        self.position.y = max(self.size, min(self.position.y, height - self.size))
        # ------------------------------------------
        
        self.knowledge += 1
        return self.knowledge > self.level * 150 

    def draw(self, screen, assets: AssetManager, offset: pygame.Vector2, player_pos: pygame.Vector2):
        draw_pos = self.position + offset
        
        angle = 0
        if player_pos:
            dx = player_pos.x - self.position.x
            dy = player_pos.y - self.position.y
            angle = math.degrees(math.atan2(dy, -dx)) + 90

        if assets.ai_image:
            rot_img = pygame.transform.rotate(assets.ai_image, angle)
            rect = rot_img.get_rect(center=(draw_pos.x, draw_pos.y))
            screen.blit(rot_img, rect)
        else:
            pygame.draw.circle(screen, self.color, (int(draw_pos.x), int(draw_pos.y)), self.size)
            
        lbl = assets.font_md.render(self.name, True, (255, 255, 255))
        screen.blit(lbl, (draw_pos.x - lbl.get_width()//2, draw_pos.y - 65))

# --- Main Game Engine ---
class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.font.init()

        self.config_mode = INITIAL_VIEW_MODE
        self.width = VIEW_CONFIGS[self.config_mode]['width']
        self.height = VIEW_CONFIGS[self.config_mode]['height']
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.scene = "menu"
        self.frame = 0
        self.delta = 0
        
        self.assets = AssetManager()
        self.assets.load_all(self.width, self.height)
        
        self.player = Player(self.width // 2, self.height // 2)
        self.ai = AI(0, 0)
        
        self.shake = 0
        self.shake_offset = pygame.Vector2(0,0)

    def set_resolution(self, mode):
        if mode in VIEW_CONFIGS:
            self.config_mode = mode
            self.width = VIEW_CONFIGS[mode]['width']
            self.height = VIEW_CONFIGS[mode]['height']
            self.screen = pygame.display.set_mode((self.width, self.height))
            
            self.assets.generate_procedural_assets(self.width, self.height)
            
            self.player.reset(self.width // 2, self.height // 2)
            self.ai.position = pygame.Vector2(0, 0)
            print(f"Resolution changed to {mode}")

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.scene == "play": self.scene = "menu"
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                self.handle_clicks(mx, my)

            if self.scene == "play":
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_a, pygame.K_LEFT]:  self.player.velocity.x = -1
                    if event.key in [pygame.K_d, pygame.K_RIGHT]: self.player.velocity.x = 1
                    if event.key in [pygame.K_w, pygame.K_UP]:    self.player.velocity.y = -1
                    if event.key in [pygame.K_s, pygame.K_DOWN]:  self.player.velocity.y = 1
                    if event.key == pygame.K_SPACE and self.player.boost_cooldown == 0:
                        self.player.boost_timer = BOOST_DURATION
                        self.player.boost_cooldown = BOOST_COOLDOWN

                if event.type == pygame.KEYUP:
                    if event.key in [pygame.K_a, pygame.K_LEFT]:  self.player.velocity.x = 0
                    if event.key in [pygame.K_d, pygame.K_RIGHT]: self.player.velocity.x = 0
                    if event.key in [pygame.K_w, pygame.K_UP]:    self.player.velocity.y = 0
                    if event.key in [pygame.K_s, pygame.K_DOWN]:  self.player.velocity.y = 0

    def handle_clicks(self, mx, my):
        cx, cy = self.width // 2, self.height // 2
        
        # Helper to check center column buttons
        btn_width = 250
        btn_x = (self.width - btn_width) // 2
        
        if self.scene == "menu":
            if pygame.Rect(btn_x, cy, 250, 60).collidepoint(mx, my):
                self.start_level(1)
            if pygame.Rect(btn_x, cy+80, 250, 60).collidepoint(mx, my):
                self.scene = "settings"
                
        elif self.scene == "settings":
            # Re-calculate specific settings layout
            if pygame.Rect(cx-150, cy-60, 300, 50).collidepoint(mx, my):
                self.set_resolution('desktop')
            if pygame.Rect(cx-150, cy, 300, 50).collidepoint(mx, my):
                self.set_resolution('mobile')
            if pygame.Rect(cx-150, cy+70, 60, 50).collidepoint(mx, my):
                self.assets.set_volume(self.assets.global_volume - 0.1)
            if pygame.Rect(cx+90, cy+70, 60, 50).collidepoint(mx, my):
                self.assets.set_volume(self.assets.global_volume + 0.1)
            if pygame.Rect(cx-150, cy+180, 300, 60).collidepoint(mx, my):
                self.scene = "menu"
                
        elif self.scene == "next_round":
            if pygame.Rect(btn_x, cy, 250, 60).collidepoint(mx, my):
                self.start_level(self.ai.level + 1)

    def start_level(self, level):
        self.player.reset(self.width // 2, self.height // 2)
        self.ai.reset(self.width, self.height, level)
        self.scene = "level_intro"
        self.wait_timer = self.frame
        
        track_num = ((level - 1) % 3) + 1
        self.assets.play_music(f"level_{track_num}.wav")

    def update(self):
        self.delta = self.clock.tick(60) / 1000.0
        self.frame += 1
        
        if self.scene == "menu":
            self.assets.play_music("menu_theme.wav")

        if self.scene == "play":
            self.player.update(self.delta, (self.width, self.height), self.assets)
            level_up = self.ai.update(self.player.position, self.delta, self.width, self.height, self.frame)
            
            if level_up:
                self.scene = "next_round"

            # --- FIXED: Smoother, Closer Shake Logic ---
            dist = self.player.position.distance_to(self.ai.position)
            radii = self.player.size + self.ai.size
            
            # Start shaking when within 200 pixels
            if dist < SHAKE_DISTANCE_THRESHOLD:
                # Calculate intensity: 0 at 200px, ramping up as distance decreases
                intensity = int((SHAKE_DISTANCE_THRESHOLD - dist) / 16)
                self.shake = min(intensity, MAX_SHAKE_INTENSITY)

                # Collision Check
                if dist < radii - 5: 
                    self.scene = "game_over"
                    self.wait_timer = self.frame
            else:
                self.shake = 0
            # -------------------------------------------

            if self.shake > 0:
                self.shake_offset = pygame.Vector2(random.randint(-self.shake, self.shake), random.randint(-self.shake, self.shake))
            else:
                self.shake_offset = pygame.Vector2(0,0)
                
        elif self.scene == "level_intro":
            if self.frame > self.wait_timer + 60: self.scene = "play"
            
        elif self.scene == "game_over":
            if self.frame > self.wait_timer + 180:
                self.scene = "menu"

    def draw(self):
        self.screen.fill((0,0,0))
        
        if self.scene == "play" or "game_over" in self.scene:
            if self.assets.tiles:
                tile = self.assets.tiles[(self.ai.level - 1) % len(self.assets.tiles)]
                tw, th = tile.get_size()
                ox, oy = int(self.shake_offset.x) % tw, int(self.shake_offset.y) % th
                for x in range(-tw, self.width, tw):
                    for y in range(-th, self.height, th):
                        self.screen.blit(tile, (x + ox, y + oy))
            
            self.ai.draw(self.screen, self.assets, self.shake_offset, self.player.position)
            self.player.draw(self.screen, self.assets, self.shake_offset)
            
            if self.assets.vignette: self.screen.blit(self.assets.vignette, (0,0))
            self.draw_hud()

        if self.scene == "menu": self.draw_menu()
        if self.scene == "settings": self.draw_settings()
        if self.scene == "level_intro": self.draw_level_intro()
        if self.scene == "next_round": self.draw_next_round()
        if self.scene == "game_over": self.draw_game_over()
        
        pygame.display.flip()

    def draw_hud(self):
        ui_x, ui_y = self.width - 250, 10
        s = pygame.Surface((240, 100), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        pygame.draw.rect(s, (50, 50, 50), s.get_rect(), 2)
        self.screen.blit(s, (ui_x, ui_y))
        
        loss = sum(self.ai.loss_history[-50:])/50 if self.ai.loss_history else 0
        self.screen.blit(self.assets.font_mono.render(f"Score: {self.frame//6}", True, (255,255,255)), (ui_x+10, ui_y+10))
        self.screen.blit(self.assets.font_mono.render(f"AI Loss: {loss:.4f}", True, (200,200,200)), (ui_x+10, ui_y+40))
        
        bx, by = ui_x+10, ui_y+70
        pygame.draw.rect(self.screen, (50,50,50), (bx, by, 220, 15))
        if self.player.boost_timer > 0:
            pygame.draw.rect(self.screen, (0,255,255), (bx, by, 220, 15))
            lbl = "BOOSTING"
        elif self.player.boost_cooldown > 0:
            pct = 1 - (self.player.boost_cooldown / BOOST_COOLDOWN)
            pygame.draw.rect(self.screen, (255,140,0), (bx, by, 220*pct, 15))
            lbl = "RECHARGING"
        else:
            pygame.draw.rect(self.screen, (0,255,0), (bx, by, 220, 15))
            lbl = "READY"
        
        txt = self.assets.font_sm.render(lbl, True, (0,0,0) if self.player.boost_timer <=0 and self.player.boost_cooldown <=0 else (255,255,255))
        self.screen.blit(txt, (bx + 110 - txt.get_width()//2, by - 2))

    def draw_btn(self, rect, text, col_base, col_hov):
        mx, my = pygame.mouse.get_pos()
        hover = rect.collidepoint(mx, my)
        pygame.draw.rect(self.screen, col_hov if hover else col_base, rect)
        lbl = self.assets.font_md.render(text, True, (255,255,255))
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def draw_menu(self):
        # Draw Background
        if self.assets.title_bg:
            bg = self.assets.title_bg
            bg_rect = bg.get_rect()
            scale = max(self.width / bg_rect.width, self.height / bg_rect.height)
            nw, nh = int(bg_rect.width * scale), int(bg_rect.height * scale)
            img = pygame.transform.scale(bg, (nw, nh))
            self.screen.blit(img, ((self.width - nw)//2, (self.height - nh)//2))
        else:
            self.screen.fill("black")

        # Center Column Layout (max width 300)
        btn_width = 250
        col_x = (self.width - btn_width) // 2
        cy = self.height // 2
        
        # Buttons constrained to center
        self.draw_btn(pygame.Rect(col_x, cy, btn_width, 60), "Start Game", (0,150,0), (0,200,0))
        self.draw_btn(pygame.Rect(col_x, cy + 80, btn_width, 60), "Settings", (50,50,150), (80,80,200))

    def draw_settings(self):
        cx, cy = self.width // 2, self.height // 2
        
        overlay = pygame.Surface((self.width, self.height))
        overlay.fill("black")
        self.screen.blit(overlay, (0,0))
        
        t = self.assets.font_lg.render("Settings", True, (255,255,255))
        self.screen.blit(t, t.get_rect(center=(cx, self.height//4)))
        
        self.draw_btn(pygame.Rect(cx-150, cy-60, 300, 50), "Desktop Mode", (0,100,100), (0,150,150))
        self.draw_btn(pygame.Rect(cx-150, cy, 300, 50), "Mobile Mode", (100,100,0), (150,150,0))
        
        self.draw_btn(pygame.Rect(cx-150, cy+70, 60, 50), "-", (100,50,50), (150,50,50))
        vol_txt = self.assets.font_md.render(f"Vol: {int(self.assets.global_volume*100)}%", True, (255,255,255))
        self.screen.blit(vol_txt, vol_txt.get_rect(center=(cx, cy+95)))
        self.draw_btn(pygame.Rect(cx+90, cy+70, 60, 50), "+", (50,100,50), (50,150,50))
        
        self.draw_btn(pygame.Rect(cx-150, cy+180, 300, 60), "Back", (150,50,50), (200,80,80))

    def draw_next_round(self):
        self.screen.fill((20,20,30))
        t = self.assets.font_lg.render(f"Level {self.ai.level} Complete", True, (0,255,255))
        self.screen.blit(t, t.get_rect(center=(self.width//2, self.height//3)))
        
        # Center the Next Round button too
        btn_width = 250
        col_x = (self.width - btn_width) // 2
        self.draw_btn(pygame.Rect(col_x, self.height//2, btn_width, 60), "Next Round", (0,100,200), (0,150,250))

    def draw_level_intro(self):
        self.screen.fill((0,0,0))
        t = self.assets.font_xl.render(str(self.ai.level), True, (255,255,255))
        self.screen.blit(t, t.get_rect(center=(self.width//2, self.height//2)))

    def draw_game_over(self):
        s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        s.fill((50, 0, 0, 100))
        self.screen.blit(s, (0,0))
        t = self.assets.font_lg.render("Humanity Lost", True, (255,0,0))
        self.screen.blit(t, t.get_rect(center=(self.width//2, self.height//2)))
        self.draw_hud()

if __name__ == "__main__":
    game = Game()
    while game.running:
        game.handle_input()
        game.update()
        game.draw()
    pygame.quit()