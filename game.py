# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable, Tuple, List

import pygame
import pygame.font
import torch
import numpy as np

# --- Configuration Constants ---
VIEW_CONFIGS = {
    'desktop': {'width': 1024, 'height': 768},
    'mobile':  {'width': 450, 'height': 800},
}
# Initial view mode and configuration (used before the game object is created)
INITIAL_VIEW_MODE = 'desktop'
INITIAL_CONFIG = VIEW_CONFIGS[INITIAL_VIEW_MODE]
INITIAL_WIDTH = INITIAL_CONFIG['width']
INITIAL_HEIGHT = INITIAL_CONFIG['height']

PLAYER_NAMES = [
    'QuantifiedQuantum',
    'Kalamata',
    'EmoAImusic',
    'MD',
    'Torva',
    'Haidar',
    'BoboBear',
    'Mohamed',
    'Alucard',
    'Kevin',
    'Barry',
    'Uniqueux',
    'JanHoleman',
    'TheJAM',
    'megansub',
    'Dereck',
    'Kyle',
    'Tuleku',
    'Travis',
    'Valor',
    'Lukey',
    'Mosh',
    'Alazr',
    'Ahmed',
]

## AI Names
AI_NAMES = [
    'HAL9000',
    'Skynet',
    'Skynet',
    'Predator',
    'DeepBlue',
    'AlphaGo',
    'Watson',
    'Siri',
    'nAIma',
    'Aldan',
    'mAIa',
    'nAlma',
    'gAIl',
    'bAIley',
    'dAIsy',
]

pygame.init()
pygame.font.get_init()
pygame.mixer.init()
pygame.mixer.music.load("ai.wav")
pygame.mixer.music.set_volume(0.01)
pygame.mixer.music.play(-1)

## TODO: STORY?
## TODO: why is the AI chasing the player? THOR AND HAMMER?
## TODO: 
## TODO: Max level for winner!!!! 20 levels
## TODO:  Imporve font management and a font class
## TODO: 
## TODO:  POINT SYSTEM to see how good you did!
## TODO: 
## TODO: 
## TODO:  BOSS FIGHT AFTER NUMBER OF LEVELS
## TODO: 
## TODO: 
## TODO: 
## TODO: Multiplayer?!?!?!? 
## TODO: number of active players
## TODO: 
## TODO:
## TODO: special abilities ( extra hits, faster )
## TODO: 
## TODO: enemy capabile of fireing weapon
## TODO: 
## TODO: instead of cirlcles somethign ELSE!!!
## TODO:   human / robot / emoji ( and different abilities )
## TODO: 
## TODO: square lis like a riot shirled where it gest random boosts
## TODO: 
## TODO: Ideas
## TODO: when you player is captured, next player can be saved
## TODO: intro a toruch that spreads as visual range like a trianagle as a torch light
## TODO: WALLS
## TODO:    like rando blocks to hide behind
## TODO: as AI gets closer to player add transparent PULSES PULSES PULSES
## TODO: 
## TODO: 
## TODO: add places to hide
## TODO: name the game

## Game Name Ideas
## -------------------
## AI Terminator
## ChAIse
## Jury Duty 5
## Revenge of The AI IRS
## Stalker
## Wanted
## You Got AI Mail
## City of Dis
##


@dataclass
class Entity:
    name: str
    position: pygame.Vector2
    speed: float
    size: int
    color: Tuple[int, int, int]
    text_color: Tuple[int, int, int]

@dataclass
class Wall(Entity):
    pass

@dataclass
class AI(Entity):
    level: int
    model: None
    learning: int
    # Corrected type hints
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

@dataclass
class Background():
    color: Tuple[int, int, int]
    shake: int
    rect: pygame.Rect

## Main Game State
@dataclass
class Game:
    player: Player
    ai: AI
    background: Background
    running: bool
    width: int
    height: int
    scene: str # menu, settings, game over, play
    delta: int
    frame: int
    wait_frame: int
    font: pygame.font.Font
    status_font: pygame.font.Font
    game_over_font: pygame.font.Font
    big_font: pygame.font.Font
    screen: pygame.surface.Surface
    clock: pygame.time.Clock
    view_mode: str # 'desktop' or 'mobile'


## Game Resolution (Initial Setup)
width  = INITIAL_WIDTH
height = INITIAL_HEIGHT


## Main Game Setup
player = Player(
    score=1,
    name=np.random.choice(PLAYER_NAMES),
    position=pygame.Vector2(1, 1),
    speed=0,
    size=30,
    color=(0, 255, 255),
    text_color=(0, 155, 155),
)
## TODO: reduce modle for lower levels
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
    learning=5, # 1 is highest, 10 is lowest
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
    background=Background(
        color=(100, 0, 250),
        shake=0,
        rect=pygame.Rect(0,0,width,height)
    ),
    font=pygame.font.SysFont("Arial", 26),
    status_font=pygame.font.SysFont("Mono", 36),
    game_over_font=pygame.font.SysFont("Arial", 50),
    big_font=pygame.font.SysFont("Arial", 300),
    width=width,
    height=height,
    running=True,
    scene="menu", # Start in the menu
    delta=0,
    frame=0,
    wait_frame=0,
    screen=pygame.display.set_mode((width, height)),
    clock=pygame.time.Clock(),
    view_mode=INITIAL_VIEW_MODE
)

# --- Helper Functions for UI ---

def draw_button(surface, rect, text, font, base_color, hover_color, mouse_click):
    """Draws a button and returns True if clicked."""
    mouse_pos = pygame.mouse.get_pos()
    is_hovering = rect.collidepoint(mouse_pos)
    color = hover_color if is_hovering else base_color

    pygame.draw.rect(surface, color, rect)
    
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)
    
    # Return True only if hovering AND a click event was registered
    return is_hovering and mouse_click

def set_screen_size(game: Game, new_mode: str):
    """Updates the game state and display surface with new size."""
    if new_mode in VIEW_CONFIGS:
        config = VIEW_CONFIGS[new_mode]
        game.width = config['width']
        game.height = config['height']
        game.view_mode = new_mode
        # Re-initialize the screen with the new size
        game.screen = pygame.display.set_mode((game.width, game.height))
        # Update the background rect to match new screen size
        game.background.rect = pygame.Rect(0, 0, game.width, game.height)
        # Update AI position to be centered on the new screen
        game.ai.position.x = game.width // 2
        game.ai.position.y = game.height * 2 # Off-screen start

# --- Scene Rendering Functions ---

def render_menu_scene(game: Game, mouse_click: bool): # UPDATED: Accepts mouse_click
    game.screen.fill("black")
    
    # Title Text
    title_text = game.game_over_font.render("Revenge of The AI IRS", True, (255, 255, 255))
    game.screen.blit(title_text, (game.width // 2 - title_text.get_width() // 2, game.height // 4))

    # --- Buttons Setup ---
    button_height = 60
    button_width = 250
    center_x = game.width // 2
    
    # 1. Start Button
    start_rect = pygame.Rect(center_x - button_width // 2, game.height // 2, button_width, button_height)
    if draw_button(game.screen, start_rect, "Start Game", game.font, (0, 150, 0), (0, 200, 0), mouse_click): # UPDATED: Pass mouse_click
        game.wait_frame = game.frame
        game.scene = "play"
        set_level(game, level=1) # Start Level 1

    # 2. Settings Button
    settings_rect = pygame.Rect(center_x - button_width // 2, game.height // 2 + button_height + 20, button_width, button_height)
    if draw_button(game.screen, settings_rect, "Settings", game.font, (50, 50, 150), (80, 80, 200), mouse_click): # UPDATED: Pass mouse_click
        game.scene = "settings"
    
    pygame.display.flip()


def render_settings_scene(game: Game, mouse_click: bool): # UPDATED: Accepts mouse_click
    game.screen.fill("black")
    
    # Title Text
    title_text = game.game_over_font.render("Settings", True, (255, 255, 255))
    game.screen.blit(title_text, (game.width // 2 - title_text.get_width() // 2, game.height // 4))

    # --- Buttons Setup ---
    button_height = 60
    button_width = 300
    center_x = game.width // 2
    y_start = game.height // 2
    
    # 1. Desktop View Button
    desktop_text = f"Desktop View ({VIEW_CONFIGS['desktop']['width']}x{VIEW_CONFIGS['desktop']['height']})"
    desktop_rect = pygame.Rect(center_x - button_width // 2, y_start, button_width, button_height)
    if draw_button(game.screen, desktop_rect, desktop_text, game.font, (0, 100, 100), (0, 150, 150), mouse_click): # UPDATED: Pass mouse_click
        set_screen_size(game, 'desktop')
        game.scene = "menu" # Return to main menu

    # 2. Mobile View Button
    mobile_text = f"Mobile View ({VIEW_CONFIGS['mobile']['width']}x{VIEW_CONFIGS['mobile']['height']})"
    mobile_rect = pygame.Rect(center_x - button_width // 2, y_start + button_height + 20, button_width, button_height)
    if draw_button(game.screen, mobile_rect, mobile_text, game.font, (100, 100, 0), (150, 150, 0), mouse_click): # UPDATED: Pass mouse_click
        set_screen_size(game, 'mobile')
        game.scene = "menu" # Return to main menu
        
    # 3. Back Button (Always needed in settings)
    back_rect = pygame.Rect(center_x - button_width // 2, y_start + (button_height + 20) * 2, button_width, button_height)
    if draw_button(game.screen, back_rect, "Back to Menu", game.font, (150, 50, 50), (200, 80, 80), mouse_click): # UPDATED: Pass mouse_click
        game.scene = "menu"
        
    pygame.display.flip()

# --- Existing Game Functions (Condensed) ---

def set_level(game: Game, level: int):
    game.player.name   = np.random.choice(PLAYER_NAMES)
    game.ai.name       = np.random.choice(AI_NAMES)

    game.ai.level      = level
    game.ai.knowledge  = 0
    game.ai.speed      = 1 + (level * 0.4)
    game.ai.size       = 40 + (level * 2)
    game.ai.position.x = game.width // 2 
    game.ai.position.y = game.height * 2

def collision(game: Game):
    distance = proximity(
        ai.position.x,
        ai.position.y,
        player.position.x,
        player.position.y,
    )
    radi = ai.size + player.size
    if distance <= radi:
        return True, distance - radi
    else:
        return False, distance - radi

def proximity(x1, y1, x2, y2):
    xd = (x1 - x2) ** 2
    yd = (y1 - y2) ** 2
    return int(np.sqrt(xd + yd))

## Render Player and AI
def render_entity(game: Game, entity: Entity):
    position = entity.position
    pygame.draw.circle(game.screen, entity.color, entity.position, entity.size)

    text = game.font.render(entity.name, True, entity.text_color)
    game.screen.blit(text, (
        position.x - text.get_width()  // 2,
        position.y - text.get_height() // 2,
    ))

def render_player(game: Game):
    x, y = pygame.mouse.get_pos()
    position = game.player.position
    position.x = x
    position.y = y

    collided, distance = collision(game)
    if 400 >= distance:
        game.background.shake = (400 - distance) // 50
        pygame.mixer.music.set_volume((400 - distance)/400)
    else:
        game.background.shake = 0
        pygame.mixer.music.set_volume(0.01)

    shake = game.background.shake
    if shake:
        position.x += np.random.randint(-shake, shake)
        position.y += np.random.randint(-shake, shake)

    render_entity(game, game.player)

    ## Cooldown invincibility while player get's to safety
    if game.frame > game.wait_frame + 100:
        collided, distance = collision(game)
        if collided:
            game.wait_frame = game.frame
            game.scene = "game over"

def render_stats(game: Game):
    if not game.scene == "game over":
        game.score = game.frame

    cost      = sum(game.ai.losses) / (len(game.ai.losses) or 1)
    output    = game.ai.output
    features  = game.ai.features
    labels    = game.ai.labels
    ai_status = f"""
AI Cost: (knowledge):  {cost:.4f} 
Features (input data): {features[0][0]:.4f} 
                       {features[0][1]:.4f} 
                       {features[0][2]:.4f} 
                       {features[0][3]:.4f} 
Labels: (training):    {labels[0][0]:.3f} 
                       {labels[0][1]:.3f} 
Output: (AI movement): {output[0][0]:.3f} 
                       {output[0][1]:.3f} 
Score: (Player):       {game.score // 6}
    """
    text = game.status_font.render(ai_status, True, (128,128,128))
    game.screen.blit(text, (game.width - 10 - text.get_width(), 10))

def render_ai(game: Game):
    ## Calculate slope for both x and y for AI to Player
    dx = game.player.position.x - game.ai.position.x
    dy = game.player.position.y - game.ai.position.y

    ## Normalize as well
    width  = game.width
    height = game.height
    labels   = [[dx/width, dy/height]]
    features = [[
        game.player.position.x / width,
        game.player.position.y / height,
        game.ai.position.x     / width,
        game.ai.position.y     / height,
    ]]
    game.ai.features = features
    game.ai.labels = labels

    ## Train and get latest directions
    ai_directions = train(game, features, labels)

    game.ai.position.x += (width/2)  * ai_directions[0][0] * game.delta * game.ai.speed
    game.ai.position.y += (height/2) * ai_directions[0][1] * game.delta * game.ai.speed

    ## Wrap around screen edges
    if game.ai.position.x < 0:
        game.ai.position.x = game.width
    if game.ai.position.x > game.width:
        game.ai.position.x = 0
    if game.ai.position.y < 0:
        game.ai.position.y = game.height
    if game.ai.position.y > game.height:
        game.ai.position.y = 0

    render_entity(game, game.ai)

def render_background(game: Game):
    game.screen.fill("black")
    shake = game.background.shake
    if shake:
        game.background.rect.x = np.random.randint(0, shake)
        game.background.rect.y = np.random.randint(0, shake)
        game.background.rect.width = game.width - np.random.randint(0, shake)
        game.background.rect.height = game.height - np.random.randint(0, shake)
    else:
        game.background.rect.x = 0
        game.background.rect.y = 0
        game.background.rect.width = game.width
        game.background.rect.height = game.height
    
    color = min(shake * 20, 255)
    bgcolor = (color, 255-color, max(0, 100-color))
    pygame.draw.rect(game.screen, bgcolor, game.background.rect)

def train(game: Game, features, labels):
    game.ai.optimizer.zero_grad()

    features = torch.tensor(features, dtype=torch.float32)
    labels   = torch.tensor(labels, dtype=torch.float32)
    output   = game.ai.model(features)

    ## For display later
    game.ai.output = output

    game.ai.knowledge += 1
    if game.ai.knowledge > game.ai.level * 150:
        set_level(game, game.ai.level + 1)
        game.wait_frame = game.frame
        game.scene = "level"
        return output

    ## TODO based on level?
    ## Train once every 5 frames
    if game.frame % game.ai.learning > 0:
        return output

    loss = game.ai.loss(output, labels)
    loss.backward()
    game.ai.optimizer.step()

    ## TODO REMOVE
    game.ai.losses.append(loss.item())
    game.ai.losses = game.ai.losses[-500:]

    return output

def render_game_scene(game: Game):
    render_background(game)
    render_stats(game)

    render_ai(game)
    render_player(game)

    pygame.display.flip()
    game.delta = game.clock.tick(60) / 1000

def render_game_over_scene(game: Game):
    game.screen.fill("black")
    text = game.game_over_font.render("Humanity Lost", True, "red")
    game.screen.blit(text, (game.width // 2 - text.get_width() // 2, game.height // 2 - text.get_height() // 2))
    render_stats(game)
    pygame.display.flip()

    ## Restart Game in a few seconds
    if game.frame > game.wait_frame + 1200:
        ## allow player to get away after game over
        game.frame = 0
        game.wait_frame = game.frame
        game.scene = "menu" # Return to menu
        set_level(game, level=1)

def render_level_scene(game):
    game.screen.fill("black")
    text = game.big_font.render(str(game.ai.level), True, (255, 255, 255))
    game.screen.blit(text, (game.width // 2 - text.get_width() // 2, game.height // 2 - text.get_height() // 2))
    pygame.display.flip()

    ## Start Next Level Game in a few seconds
    if game.frame > game.wait_frame + 500:
        game.wait_frame = game.frame
        game.scene = "play"

## Main Game Loop
while game.running:
    # --- NEW: Track Mouse Click ---
    mouse_click = False 
    
    ## Handle Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.running = False
        
        # Check for a mouse button press down event
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_click = True # Set flag if a click occurred
            
        # Check for key presses outside the menu state for debugging/quick actions
        if game.scene == "play" and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game.scene = "menu" # Quick exit to menu

    ## Game State Management
    game.frame += 1
    match game.scene:
        case "menu":
            render_menu_scene(game, mouse_click) # Pass mouse_click
            
        case "settings":
            render_settings_scene(game, mouse_click) # Pass mouse_click

        case "level":
            render_level_scene(game)

        case "play":
            render_game_scene(game)

        case "game over":
            render_game_over_scene(game)

pygame.quit()