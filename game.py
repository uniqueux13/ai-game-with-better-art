# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Dict
import math 
import os # Required to check if music files exist

import pygame
import pygame.font
import torch
import numpy as np

# --- Configuration Constants ---
VIEW_CONFIGS = {
    'desktop': {'width': 1024, 'height': 768},
    'mobile':  {'width': 450, 'height': 800},
}
INITIAL_VIEW_MODE = 'desktop'
INITIAL_CONFIG = VIEW_CONFIGS[INITIAL_VIEW_MODE]
INITIAL_WIDTH = INITIAL_CONFIG['width']
INITIAL_HEIGHT = INITIAL_CONFIG['height']

PLAYER_SPEED = 350 
BOOST_SPEED = 900  
BOOST_DURATION = 15 
BOOST_COOLDOWN = 120 
ANIMATION_SPEED = 10 

PLAYER_NAMES = [
    'QuantifiedQuantum', 'Kalamata', 'EmoAImusic', 'MD', 'Torva', 'Haidar', 
    'BoboBear', 'Mohamed', 'Alucard', 'Kevin', 'Barry', 'Uniqueux', 
    'JanHoleman', 'TheJAM', 'megansub', 'Dereck', 'Kyle', 'Tuleku', 
    'Travis', 'Valor', 'Lukey', 'Mosh', 'Alazr', 'Ahmed',
]

AI_NAMES = [
    'HAL9000', 'Skynet', 'Predator', 'DeepBlue', 'AlphaGo', 'Watson', 
    'Siri', 'nAIma', 'Aldan', 'mAIa', 'nAlma', 'gAIl', 'bAIley', 'dAIsy',
]

pygame.init()
pygame.font.get_init()
pygame.mixer.init()

# --- Asset Loading ---
PLAYER_ANIMATIONS: Dict[str, List[pygame.Surface]] = {}
BACKGROUND_TILES: List[pygame.Surface] = [] 
AI_IMAGE = None
VIGNETTE_SURFACE = None
CURRENT_TRACK = None # Helper to track what is currently playing

def play_music(track_name):
    """
    Handles music playback. 
    Prevents restarting the song if it's already playing.
    Handles looping level tracks (1 -> 2 -> 3 -> 1).
    """
    global CURRENT_TRACK
    
    # Logic: If requesting a level track, handle the 1-3 cycle
    filename = track_name
    if track_name.startswith("level_"):
        try:
            # Extract number, e.g., "level_5" -> 5
            lvl = int(track_name.split("_")[1])
            # Cycle math: ((5-1) % 3) + 1 = 2. So Level 5 plays track 2.
            cycle = ((lvl - 1) % 3) + 1 
            filename = f"level_{cycle}.wav"
        except:
            filename = "level_1.wav" # Fallback
    
    # Don't restart if already playing
    if CURRENT_TRACK == filename:
        return 

    # Load and Play
    if os.path.exists(filename):
        print(f"--> Playing Track: {filename}")
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.set_volume(0.3) # Set volume (0.0 to 1.0)
            pygame.mixer.music.play(-1) # Loop forever
            CURRENT_TRACK = filename
        except Exception as e:
            print(f"Music Load Error: {e}")
    else:
        print(f"Music file missing: {filename}")

def load_spritesheet(filename, frame_count, scale_to=(64, 64)):
    frames = []
    try:
        sheet = pygame.image.load(filename).convert_alpha()
        sheet_width, sheet_height = sheet.get_size()
        sprite_size = sheet_height 
        for i in range(frame_count):
            rect = pygame.Rect(i * sprite_size, 0, sprite_size, sprite_size)
            image = sheet.subsurface(rect)
            image = pygame.transform.scale(image, scale_to)
            frames.append(image)
        return frames
    except pygame.error as e:
        print(f"Error loading {filename}: {e}")
        fallback = pygame.Surface(scale_to)
        fallback.fill((0, 255, 255))
        return [fallback]

def generate_default_tiles():
    tiles = []
    
    # Tile 0: Tech Grid 
    s0 = pygame.Surface((64, 64))
    s0.fill((20, 30, 40))
    pygame.draw.rect(s0, (30, 50, 60), (0, 0, 64, 64), 1)
    tiles.append(s0)

    # Tile 1: Mars Ground
    s1 = pygame.Surface((64, 64))
    s1.fill((60, 30, 20))
    for i in range(10):
        x, y = np.random.randint(0, 64), np.random.randint(0, 64)
        pygame.draw.circle(s1, (50, 25, 15), (x, y), 3)
    tiles.append(s1)

    # Tile 2: Industrial Floor
    s2 = pygame.Surface((64, 64))
    s2.fill((50, 50, 50))
    pygame.draw.circle(s2, (30, 30, 30), (4, 4), 2)
    pygame.draw.circle(s2, (30, 30, 30), (60, 4), 2)
    pygame.draw.circle(s2, (30, 30, 30), (4, 60), 2)
    pygame.draw.circle(s2, (30, 30, 30), (60, 60), 2)
    tiles.append(s2)

    # Tile 3: Cyber Grass
    s3 = pygame.Surface((64, 64))
    s3.fill((10, 40, 20))
    pygame.draw.line(s3, (20, 60, 30), (0, 0), (64, 64), 1)
    pygame.draw.line(s3, (20, 60, 30), (64, 0), (0, 64), 1)
    tiles.append(s3)
    
    return tiles

def generate_vignette(width, height):
    vignette = pygame.Surface((width, height), pygame.SRCALPHA)
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    alpha_map = (dist - 0.4).clip(0, 1) * 255
    alpha_map = np.minimum(alpha_map * 1.5, 255) 
    pixels = pygame.surfarray.pixels_alpha(vignette)
    pixels[:] = alpha_map.astype(np.uint8).T 
    del pixels 
    return vignette

def load_assets():
    global PLAYER_ANIMATIONS, AI_IMAGE, BACKGROUND_TILES, VIGNETTE_SURFACE
    print("Loading sprites...")
    PLAYER_ANIMATIONS['idle_down']  = load_spritesheet("idle_down.png", 2)
    PLAYER_ANIMATIONS['idle_up']    = load_spritesheet("idle_up.png", 2)
    PLAYER_ANIMATIONS['idle_left']  = load_spritesheet("idle_left.png", 4)
    PLAYER_ANIMATIONS['idle_right'] = load_spritesheet("idle_right.png", 4)
    
    PLAYER_ANIMATIONS['walk_down']  = load_spritesheet("walk_down.png", 4)
    PLAYER_ANIMATIONS['walk_up']    = load_spritesheet("walk_up.png", 4)
    PLAYER_ANIMATIONS['walk_left']  = load_spritesheet("walk_left.png", 4)
    PLAYER_ANIMATIONS['walk_right'] = load_spritesheet("walk_right.png", 4)

    try:
        AI_IMAGE = pygame.image.load("ai_sprite.png").convert_alpha()
        AI_IMAGE = pygame.transform.scale(AI_IMAGE, (80, 80)) 
    except:
        pass

    try:
        BACKGROUND_TILES.extend(generate_default_tiles())
    except Exception as e:
        print(f"Tile generation error: {e}")

    VIGNETTE_SURFACE = generate_vignette(INITIAL_WIDTH, INITIAL_HEIGHT)

# --- Game Classes ---
@dataclass
class Entity:
    name: str
    position: pygame.Vector2
    speed: float
    size: int
    color: Tuple[int, int, int]
    text_color: Tuple[int, int, int]

@dataclass
class AI(Entity):
    level: int
    model: None
    learning: int
    optimizer: torch.optim.AdamW
    loss: torch.nn.MSELoss
    losses: List
    features: List
    labels: List
    output: List
    knowledge: int

@dataclass
class Player(Entity):
    score: float
    velocity: pygame.Vector2 = pygame.Vector2(0, 0)
    boost_timer: int = 0    
    boost_cooldown: int = 0 
    trail: List[pygame.Vector2] = field(default_factory=list)
    facing: str = 'down'      
    anim_state: str = 'idle'  
    frame_index: float = 0.0

@dataclass
class Background():
    current_tile_idx: int 
    shake: int
    offset: pygame.Vector2

@dataclass
class Game:
    player: Player
    ai: AI
    background: Background
    running: bool
    width: int
    height: int
    scene: str 
    delta: int
    frame: int
    wait_frame: int
    font: pygame.font.Font
    status_font: pygame.font.Font
    ui_font: pygame.font.Font
    game_over_font: pygame.font.Font
    big_font: pygame.font.Font
    screen: pygame.surface.Surface
    clock: pygame.time.Clock
    view_mode: str 

# --- Initialization ---
width  = INITIAL_WIDTH
height = INITIAL_HEIGHT

player = Player(
    score=1,
    name=np.random.choice(PLAYER_NAMES),
    position=pygame.Vector2(INITIAL_WIDTH // 2, INITIAL_HEIGHT // 2), 
    speed=0,
    size=30, 
    color=(0, 255, 255),
    text_color=(255, 255, 0),
    boost_timer=0,
    boost_cooldown=0,
    trail=[],
    facing='down',
    anim_state='idle',
    frame_index=0
)

model = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2),
    torch.nn.Tanh(),
)

ai = AI(
    name=np.random.choice(AI_NAMES),
    position=pygame.Vector2(width // 2, height // 2),
    level=1,
    speed=1,
    size=40, 
    color=(255, 0, 0),
    text_color=(255, 255, 255),
    model=model,
    learning=5, 
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    loss=torch.nn.MSELoss(),
    losses=[],
    features=[[0,0,0,0]],
    labels=[[0,0]],
    output=[[0,0]],
    knowledge=0,
)

game = Game(
    player=player,
    ai=ai,
    background=Background(current_tile_idx=0, shake=0, offset=pygame.Vector2(0,0)),
    font=pygame.font.SysFont("Arial", 26),
    status_font=pygame.font.SysFont("Mono", 24),
    ui_font=pygame.font.SysFont("Arial", 20, bold=True),
    game_over_font=pygame.font.SysFont("Arial", 50),
    big_font=pygame.font.SysFont("Arial", 300),
    width=width,
    height=height,
    running=True,
    scene="menu",
    delta=0,
    frame=0,
    wait_frame=0,
    screen=pygame.display.set_mode((width, height)),
    clock=pygame.time.Clock(),
    view_mode=INITIAL_VIEW_MODE
)

load_assets()

# --- Helper Functions ---
def rot_center(image, angle, x, y):
    rot_image = pygame.transform.rotate(image, angle)
    new_rect = rot_image.get_rect(center=image.get_rect(center=(x, y)).center)
    return rot_image, new_rect

def draw_button(surface, rect, text, font, base_color, hover_color, mouse_click):
    mouse_pos = pygame.mouse.get_pos()
    is_hovering = rect.collidepoint(mouse_pos)
    color = hover_color if is_hovering else base_color
    pygame.draw.rect(surface, color, rect)
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)
    return is_hovering and mouse_click

def set_screen_size(game: Game, new_mode: str):
    global VIGNETTE_SURFACE
    if new_mode in VIEW_CONFIGS:
        config = VIEW_CONFIGS[new_mode]
        game.width = config['width']
        game.height = config['height']
        game.view_mode = new_mode
        game.screen = pygame.display.set_mode((game.width, game.height))
        game.player.position = pygame.Vector2(game.width // 2, game.height // 2)
        game.ai.position = pygame.Vector2(0, 0)
        VIGNETTE_SURFACE = generate_vignette(game.width, game.height)

# --- Core Game Logic ---
def set_level(game: Game, level: int):
    game.player.name = np.random.choice(PLAYER_NAMES)
    game.ai.name = np.random.choice(AI_NAMES)
    game.ai.level = level
    game.ai.knowledge = 0
    game.ai.speed = 1 + (level * 0.4)
    game.ai.size = 40 + (level * 2)
    
    if BACKGROUND_TILES:
        game.background.current_tile_idx = (level - 1) % len(BACKGROUND_TILES)

    # UPDATED: Trigger Level Music
    # "level_1", "level_2", "level_3" ... handled by play_music logic
    play_music(f"level_{level}")

    game.player.position = pygame.Vector2(game.width // 2, game.height // 2)
    game.player.trail = [] 
    game.player.boost_cooldown = 0
    game.player.boost_timer = 0

    corners = [
        pygame.Vector2(0, 0),
        pygame.Vector2(game.width, 0),
        pygame.Vector2(0, game.height),
        pygame.Vector2(game.width, game.height)
    ]
    rand_idx = np.random.randint(0, 4)
    game.ai.position = corners[rand_idx].copy()

def collision(game: Game):
    distance = int(np.sqrt((ai.position.x - player.position.x)**2 + (ai.position.y - player.position.y)**2))
    radi = ai.size + player.size
    return (distance <= radi, distance - radi)

def render_entity_simple(game: Game, entity: Entity, image: pygame.Surface, angle: float = 0):
    if image:
        rotated_image, new_rect = rot_center(image, angle, entity.position.x, entity.position.y)
        game.screen.blit(rotated_image, new_rect)
    else:
        pygame.draw.circle(game.screen, entity.color, entity.position, entity.size)
    
    text = game.font.render(entity.name, True, entity.text_color)
    game.screen.blit(text, (entity.position.x - text.get_width() // 2, entity.position.y - 65))

def render_player(game: Game):
    p = game.player
    
    if p.boost_timer > 0:
        p.boost_timer -= 1
        current_speed = BOOST_SPEED
        p.trail.append(p.position.copy())
    else:
        current_speed = PLAYER_SPEED
        if p.trail: p.trail.pop(0)

    if len(p.trail) > 10: p.trail.pop(0)
    if p.boost_cooldown > 0: p.boost_cooldown -= 1

    move_vec = pygame.Vector2(0, 0)
    if p.velocity.length() > 0:
        move_vec = p.velocity.normalize() * current_speed
        if abs(p.velocity.x) > abs(p.velocity.y):
            p.facing = 'right' if p.velocity.x > 0 else 'left'
        else:
            p.facing = 'down' if p.velocity.y > 0 else 'up'
        p.anim_state = 'walk'
    else:
        p.anim_state = 'idle'
    
    if p.boost_timer > 0 and p.velocity.length() == 0:
        if p.facing == 'right': move_vec.x = current_speed
        elif p.facing == 'left': move_vec.x = -current_speed
        elif p.facing == 'down': move_vec.y = current_speed
        elif p.facing == 'up': move_vec.y = -current_speed

    p.position += move_vec * game.delta
    pr = p.size
    p.position.x = max(pr, min(p.position.x, game.width - pr))
    p.position.y = max(pr, min(p.position.y, game.height - pr))

    collided, distance = collision(game)
    if 400 >= distance:
        calculated_shake = (400 - distance) // 100 
        game.background.shake = min(calculated_shake, 8)
    else:
        game.background.shake = 0

    game.background.offset = pygame.Vector2(0,0)
    if game.background.shake:
        sx = np.random.randint(-game.background.shake, game.background.shake)
        sy = np.random.randint(-game.background.shake, game.background.shake)
        game.background.offset = pygame.Vector2(sx, sy)
        p.position.x += sx
        p.position.y += sy

    for i, pos in enumerate(p.trail):
        trail_size = p.size * (i / len(p.trail))
        pygame.draw.circle(game.screen, (0, 200, 255), (int(pos.x), int(pos.y)), int(trail_size))

    animation_key = f"{p.anim_state}_{p.facing}"
    if animation_key not in PLAYER_ANIMATIONS:
        animation_key = 'idle_down'

    frames = PLAYER_ANIMATIONS[animation_key]
    p.frame_index += ANIMATION_SPEED * game.delta
    if p.frame_index >= len(frames):
        p.frame_index = 0
    
    current_image = frames[int(p.frame_index)]
    rect = current_image.get_rect(center=(p.position.x, p.position.y))
    game.screen.blit(current_image, rect)

    text = game.font.render(p.name, True, p.text_color)
    game.screen.blit(text, (p.position.x - text.get_width() // 2, p.position.y - 65))

    if game.frame > game.wait_frame + 100:
        if collided:
            game.wait_frame = game.frame
            game.scene = "game over"

def render_ai(game: Game):
    dx = game.player.position.x - game.ai.position.x
    dy = game.player.position.y - game.ai.position.y
    width, height = game.width, game.height
    labels = [[dx/width, dy/height]]
    features = [[game.player.position.x/width, game.player.position.y/height, game.ai.position.x/width, game.ai.position.y/height]]
    
    game.ai.features = features
    game.ai.labels = labels
    ai_directions = train(game, features, labels)

    game.ai.position.x += (width/2) * ai_directions[0][0] * game.delta * game.ai.speed
    game.ai.position.y += (height/2) * ai_directions[0][1] * game.delta * game.ai.speed

    margin = game.ai.size
    game.ai.position.x = max(margin, min(game.ai.position.x, game.width - margin))
    game.ai.position.y = max(margin, min(game.ai.position.y, game.height - margin))

    render_pos = game.ai.position + game.background.offset
    angle = math.degrees(math.atan2(dy, -dx)) + 90
    
    if AI_IMAGE:
        rotated_image, new_rect = rot_center(AI_IMAGE, angle, render_pos.x, render_pos.y)
        game.screen.blit(rotated_image, new_rect)
    else:
        pygame.draw.circle(game.screen, game.ai.color, (int(render_pos.x), int(render_pos.y)), game.ai.size)
        
    text = game.font.render(game.ai.name, True, game.ai.text_color)
    game.screen.blit(text, (render_pos.x - text.get_width() // 2, render_pos.y - 65))

def train(game: Game, features, labels):
    game.ai.optimizer.zero_grad()
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    output = game.ai.model(features)
    game.ai.output = output
    game.ai.knowledge += 1
    
    if game.ai.knowledge > game.ai.level * 150:
        game.wait_frame = game.frame
        game.scene = "next_round" 
        return output

    if game.frame % game.ai.learning > 0: return output

    loss = game.ai.loss(output, labels)
    loss.backward()
    game.ai.optimizer.step()
    game.ai.losses.append(loss.detach().item())
    game.ai.losses = game.ai.losses[-500:]
    return output

def render_stats(game: Game):
    if not game.scene == "game over": game.score = game.frame
    
    padding = 15
    ui_width = 240
    ui_height = 100
    ui_x = game.width - ui_width - 10
    ui_y = 10
    
    ui_surface = pygame.Surface((ui_width, ui_height), pygame.SRCALPHA)
    ui_surface.fill((0, 0, 0, 180)) 
    pygame.draw.rect(ui_surface, (50, 50, 50), ui_surface.get_rect(), 2)
    game.screen.blit(ui_surface, (ui_x, ui_y))

    cost = sum(game.ai.losses) / (len(game.ai.losses) or 1)
    
    score_text = game.status_font.render(f"Score: {game.score // 6}", True, (255, 255, 255))
    cost_text = game.status_font.render(f"AI Loss: {cost:.4f}", True, (200, 200, 200))
    
    game.screen.blit(score_text, (ui_x + padding, ui_y + 10))
    game.screen.blit(cost_text, (ui_x + padding, ui_y + 40))

    bar_x = ui_x + padding
    bar_y = ui_y + 70
    bar_w = ui_width - (padding * 2)
    bar_h = 15
    
    pygame.draw.rect(game.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
    
    if game.player.boost_timer > 0:
        fill_w = bar_w
        color = (0, 255, 255)
        label = "BOOST ACTIVE"
    elif game.player.boost_cooldown > 0:
        pct = 1 - (game.player.boost_cooldown / BOOST_COOLDOWN)
        fill_w = int(bar_w * pct)
        color = (255, 140, 0)
        label = "RECHARGING"
    else:
        fill_w = bar_w
        color = (0, 255, 0)
        label = "SPACE TO BOOST"
        
    pygame.draw.rect(game.screen, color, (bar_x, bar_y, fill_w, bar_h))
    
    if game.player.boost_cooldown > 0 and game.player.boost_timer == 0:
         label_surf = game.ui_font.render(label, True, (255, 255, 255))

def render_background(game: Game):
    if BACKGROUND_TILES:
        tile = BACKGROUND_TILES[game.background.current_tile_idx]
        tile_w, tile_h = tile.get_size()
        ox = game.background.offset.x % tile_w
        oy = game.background.offset.y % tile_h
        for x in range(-tile_w, game.width, tile_w):
            for y in range(-tile_h, game.height, tile_h):
                game.screen.blit(tile, (x + ox, y + oy))
    else:
        game.screen.fill("black")

# --- Scenes ---
def render_menu_scene(game: Game, mouse_click: bool):
    # UPDATED: Play Menu Music
    play_music("menu_theme.wav")

    game.screen.fill("black")
    title_text = game.game_over_font.render("CHAISE: AI Hunter", True, (255, 255, 255))
    game.screen.blit(title_text, (game.width // 2 - title_text.get_width() // 2, game.height // 4))
    
    if draw_button(game.screen, pygame.Rect(game.width//2 - 125, game.height//2, 250, 60), "Start Game", game.font, (0,150,0), (0,200,0), mouse_click):
        game.wait_frame = game.frame
        game.scene = "play"
        set_level(game, level=1)

    if draw_button(game.screen, pygame.Rect(game.width//2 - 125, game.height//2 + 80, 250, 60), "Settings", game.font, (50,50,150), (80,80,200), mouse_click):
        game.scene = "settings"
    pygame.display.flip()

def render_settings_scene(game: Game, mouse_click: bool):
    game.screen.fill("black")
    title_text = game.game_over_font.render("Settings", True, (255, 255, 255))
    game.screen.blit(title_text, (game.width // 2 - title_text.get_width() // 2, game.height // 4))
    
    controls_text_1 = game.font.render("Controls: WASD or Arrows to Move", True, (0, 200, 255))
    controls_text_2 = game.font.render("SPACEBAR to Speed Boost", True, (0, 200, 255))
    game.screen.blit(controls_text_1, (game.width // 2 - controls_text_1.get_width() // 2, game.height // 3 + 20))
    game.screen.blit(controls_text_2, (game.width // 2 - controls_text_2.get_width() // 2, game.height // 3 + 50))

    cx = game.width // 2
    if draw_button(game.screen, pygame.Rect(cx-150, game.height//2 + 20, 300, 60), f"Desktop ({VIEW_CONFIGS['desktop']['width']}x{VIEW_CONFIGS['desktop']['height']})", game.font, (0,100,100), (0,150,150), mouse_click):
        set_screen_size(game, 'desktop')
        game.scene = "menu"
        
    if draw_button(game.screen, pygame.Rect(cx-150, game.height//2 + 100, 300, 60), f"Mobile ({VIEW_CONFIGS['mobile']['width']}x{VIEW_CONFIGS['mobile']['height']})", game.font, (100,100,0), (150,150,0), mouse_click):
        set_screen_size(game, 'mobile')
        game.scene = "menu"

    if draw_button(game.screen, pygame.Rect(cx-150, game.height//2 + 180, 300, 60), "Back", game.font, (150,50,50), (200,80,80), mouse_click):
        game.scene = "menu"
    pygame.display.flip()

def render_next_round_scene(game: Game, mouse_click: bool):
    game.screen.fill((20, 20, 30)) 
    
    level_msg = game.game_over_font.render(f"Level {game.ai.level} Complete", True, (0, 255, 255))
    game.screen.blit(level_msg, (game.width // 2 - level_msg.get_width() // 2, game.height // 3))

    if draw_button(game.screen, pygame.Rect(game.width//2 - 125, game.height//2, 250, 60), "Next Round", game.font, (0,100,200), (0,150,250), mouse_click):
        set_level(game, game.ai.level + 1)
        game.wait_frame = game.frame
        game.scene = "level"

    pygame.display.flip()

def render_level_scene(game):
    game.screen.fill("black")
    text = game.big_font.render(str(game.ai.level), True, (255, 255, 255))
    game.screen.blit(text, (game.width // 2 - text.get_width() // 2, game.height // 2 - text.get_height() // 2))
    pygame.display.flip()
    if game.frame > game.wait_frame + 60: 
        game.wait_frame = game.frame
        game.scene = "play"

def render_game_scene(game: Game):
    render_background(game)
    render_ai(game)
    render_player(game)
    if VIGNETTE_SURFACE:
        game.screen.blit(VIGNETTE_SURFACE, (0, 0))
    render_stats(game)
    pygame.display.flip()
    game.delta = game.clock.tick(60) / 1000

def render_game_over_scene(game: Game):
    game.screen.fill("black")
    text = game.game_over_font.render("Humanity Lost", True, "red")
    game.screen.blit(text, (game.width // 2 - text.get_width() // 2, game.height // 2 - text.get_height() // 2))
    render_stats(game)
    pygame.display.flip()
    if game.frame > game.wait_frame + 1200:
        game.frame = 0
        game.wait_frame = game.frame
        game.scene = "menu"
        set_level(game, level=1)

# --- Main Loop ---
while game.running:
    mouse_click = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT: game.running = False
        if event.type == pygame.MOUSEBUTTONDOWN: mouse_click = True
        
        if game.scene != "play" and event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            game.scene = "menu"
            
        if game.scene == "play":
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_a, pygame.K_LEFT]:  game.player.velocity.x = -1
                if event.key in [pygame.K_d, pygame.K_RIGHT]: game.player.velocity.x = 1
                if event.key in [pygame.K_w, pygame.K_UP]:    game.player.velocity.y = -1
                if event.key in [pygame.K_s, pygame.K_DOWN]:  game.player.velocity.y = 1
                
                if event.key == pygame.K_SPACE and game.player.boost_cooldown == 0:
                    game.player.boost_timer = BOOST_DURATION
                    game.player.boost_cooldown = BOOST_COOLDOWN

            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_a, pygame.K_LEFT]:  game.player.velocity.x = 0
                if event.key in [pygame.K_d, pygame.K_RIGHT]: game.player.velocity.x = 0
                if event.key in [pygame.K_w, pygame.K_UP]:    game.player.velocity.y = 0
                if event.key in [pygame.K_s, pygame.K_DOWN]:  game.player.velocity.y = 0

    game.frame += 1
    match game.scene:
        case "menu":       render_menu_scene(game, mouse_click)
        case "settings":   render_settings_scene(game, mouse_click)
        case "next_round": render_next_round_scene(game, mouse_click)
        case "level":      render_level_scene(game)
        case "play":       render_game_scene(game)
        case "game over":  render_game_over_scene(game)

pygame.quit()