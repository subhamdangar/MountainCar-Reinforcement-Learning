import numpy as np
import pygame
import gymnasium as gym
import torch
import torch.nn as nn
import time
import math

# ================================
#  CONFIG
# ================================
WIDTH, HEIGHT = 900, 500
FPS = 40
STEP_EVERY_N_FRAMES = 2  # slow down physics

X_MIN, X_MAX = -1.2, 0.6
GOAL_X = 0.5

CAR_W = 36
CAR_H = 16
WHEEL_R = 7

# ================================
#  LOAD DQN MODEL
# ================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.model(x)

device = torch.device("cpu")
model = DQN(2, 3).to(device)
model.load_state_dict(torch.load("dqn_mountaincar.pth", map_location=device))
model.eval()
print("✅ Loaded DQN model")

# ================================
#  SOUND SETUP
# ================================
pygame.mixer.init()
engine_sound = pygame.mixer.Sound("engine.wav")
success_sound = pygame.mixer.Sound("success.wav")
engine_sound.set_volume(0.35)
success_sound.set_volume(0.8)

# ================================
#  DUST PARTICLES
# ================================
DUST_MAX = 300
DUST_LIFETIME = 1.0
DUST_COLOR = (200, 200, 200)
DUST_SIZE = 6

dust_particles = []

def emit_dust(x, vel):
    sx = x_to_screen(x)
    sy = y_to_screen(hill_y(x))
    for _ in range(3 + int(abs(vel) * 80)):
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 0.3)
        dust_particles.append([sx, sy + 10, vx, vy, 1.0])
    while len(dust_particles) > DUST_MAX:
        dust_particles.pop(0)

def update_dust(dt):
    remove = []
    for p in dust_particles:
        p[0] += p[2]
        p[1] += p[3]
        p[4] -= dt * 1.2
        if p[4] <= 0:
            remove.append(p)
    for p in remove:
        dust_particles.remove(p)

def draw_dust(screen):
    for p in dust_particles:
        alpha = max(0, int(p[4] * 255))
        s = pygame.Surface((DUST_SIZE, DUST_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(s, (*DUST_COLOR, alpha), (DUST_SIZE//2, DUST_SIZE//2), DUST_SIZE//2)
        screen.blit(s, (p[0], p[1]))

# ================================
#  GRAPHICS HELPERS
# ================================
def hill_y(x): return 0.55 + 0.45 * math.sin(3 * x)

def x_to_screen(x):
    return int((x - X_MIN) / (X_MAX - X_MIN) * (WIDTH - 100) + 50)

def y_to_screen(y):
    pad_top, pad_bottom = 40, 70
    usable = HEIGHT - pad_top - pad_bottom
    return int(HEIGHT - pad_bottom - y * usable)

clouds = [[100, 90], [350, 120], [700, 80]]

def draw_cloud(s, x, y):
    pygame.draw.ellipse(s, (245,245,255), (x, y, 80, 40))
    pygame.draw.ellipse(s, (245,245,255), (x+30, y-20, 80, 50))
    pygame.draw.ellipse(s, (245,245,255), (x+60, y, 80, 40))

def update_clouds():
    for c in clouds:
        c[0] += 0.3
        if c[0] > WIDTH + 100:
            c[0] = -100

def draw_sky(screen):
    for i in range(HEIGHT):
        t = i / HEIGHT
        r = int(120*(1-t) + 200*t)
        g = int(180*(1-t) + 220*t)
        b = int(230*(1-t) + 255*t)
        pygame.draw.line(screen, (r,g,b), (0,i), (WIDTH,i))

    update_clouds()
    for c in clouds:
        draw_cloud(screen, c[0], c[1])

def draw_hills(screen):
    pts = []
    for sx in range(WIDTH):
        x = X_MIN + (sx-50) * (X_MAX-X_MIN) / (WIDTH-100)
        pts.append((sx, y_to_screen(hill_y(x))))
    pygame.draw.polygon(screen, (60,180,90), [(0, HEIGHT)] + pts + [(WIDTH, HEIGHT)])
    pygame.draw.lines(screen, (30,120,50), False, pts, 3)

def draw_flag(screen):
    gx = x_to_screen(GOAL_X)
    gy = y_to_screen(hill_y(GOAL_X))
    pygame.draw.line(screen, (230,230,230), (gx,gy-60), (gx,gy), 4)
    wave = math.sin(time.time() * 4) * 6
    pygame.draw.polygon(screen, (255,120,40), [(gx,gy-60), (gx+35, gy-55+wave), (gx,gy-40)])

def draw_car(screen, x, vel):
    dx = 0.002
    dy = hill_y(x+dx) - hill_y(x-dx)
    angle = math.degrees(math.atan2(dy, 2*dx))

    cx = x_to_screen(x)
    cy = y_to_screen(hill_y(x))

    # Car body
    body = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
    pygame.draw.rect(body, (250,80,80), (0,CAR_H//2,CAR_W,CAR_H//2), border_radius=6)
    pygame.draw.rect(body, (255,240,120), (6,0,CAR_W-12,CAR_H//2), border_radius=6)
    body_rot = pygame.transform.rotate(body, -angle)
    screen.blit(body_rot, body_rot.get_rect(center=(cx,cy-8)))

    # Wheels
    spin = (pygame.time.get_ticks() / 80) % 360
    wheel = pygame.Surface((WHEEL_R*2, WHEEL_R*2), pygame.SRCALPHA)
    pygame.draw.circle(wheel, (20,20,20), (WHEEL_R,WHEEL_R), WHEEL_R)
    pygame.draw.line(wheel, (255,255,255), (WHEEL_R,WHEEL_R),
                     (WHEEL_R+WHEEL_R*math.cos(spin), WHEEL_R+WHEEL_R*math.sin(spin)), 3)

    for wx in [-CAR_W//3, CAR_W//3]:
        wr = pygame.transform.rotate(wheel, -angle)
        screen.blit(wr, wr.get_rect(center=(cx+wx, cy-2)))

def draw_hud(screen, step, ret):
    font = pygame.font.SysFont("Arial", 18)
    txt = f"Step: {step}   Return: {ret:.1f}   Mode: DQN"
    screen.blit(font.render(txt, True, (20,20,20)), (10,10))

# ================================
#  MAIN LOOP
# ================================
def main():
    env = gym.make("MountainCar-v0")       

    obs, _ = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    engine_sound.play(loops=-1)

    frame = 0
    step = 0
    ret = 0
    running = True

    while running:
        frame += 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # slow physics
        if frame % STEP_EVERY_N_FRAMES == 0:
            with torch.no_grad():
                a = model(torch.FloatTensor(obs)).argmax().item()

            obs, r, term, trunc, _ = env.step(a)
            ret += r
            step += 1

            # dust
            if abs(obs[1]) > 0.015:
                emit_dust(obs[0], obs[1])

            if obs[0] >= GOAL_X:
                success_sound.play()

            if term or trunc:
                dust_particles.clear()
                obs, _ = env.reset()
                step = 0
                ret = 0

        # update dust animation
        dt = clock.get_time() / 1000.0
        update_dust(dt)

        # draw
        draw_sky(screen)
        draw_hills(screen)
        draw_flag(screen)
        draw_dust(screen)
        draw_car(screen, obs[0], obs[1])
        draw_hud(screen, step, ret)

        pygame.display.flip()
        clock.tick(FPS)

    engine_sound.stop()
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
