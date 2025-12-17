import math
import sys
import time
import numpy as np
import pygame
import gymnasium as gym
pygame.mixer.init()


# -----------------------------
# CONFIG
# -----------------------------
WIDTH, HEIGHT = 900, 500
FPS = 60
ENV_ID = "MountainCar-v0"

# Environment bounds
X_MIN, X_MAX = -1.2, 0.6
GOAL_X = 0.5

# Car design
CAR_W = 36
CAR_H = 16
WHEEL_R = 7

# -------- Dust / Particles --------
DUST_MAX = 300               # cap particle count
DUST_SPAWN_SPEED = 0.015     # minimum |vel_x| to spawn dust
DUST_SPAWN_COOLDOWN = 0.02   # seconds between spawns
DUST_LIFETIME = 0.9          # seconds particle lasts
DUST_BASE_SIZE = 6
DUST_COLOR = (200, 200, 200)
dust_particles = []
_last_dust_spawn = 0.0       # used to throttle spawn rate


# RL discretization (for loading Q-table)
POS_BINS = 24
VEL_BINS = 20
pos_space = np.linspace(X_MIN, X_MAX, POS_BINS - 1)
vel_space = np.linspace(-0.07, 0.07, VEL_BINS - 1)

def discretize(obs):
    pos, vel = obs
    pi = np.digitize(pos, pos_space)
    vi = np.digitize(vel, vel_space)
    return pi, vi

# -----------------------------
# MATH / HILL FUNCTIONS
# -----------------------------
def hill_y(x):
    return 0.55 + 0.45 * math.sin(3.0 * x)

def x_to_screen(x):
    return int((x - X_MIN) / (X_MAX - X_MIN) * (WIDTH - 100) + 50)

def y_to_screen(y_norm):
    pad_top, pad_bottom = 40, 70
    usable = HEIGHT - pad_top - pad_bottom
    return int(HEIGHT - pad_bottom - y_norm * usable)

def env_to_screen_xy(x):
    """Given env x, return (screen_x, screen_y) placed on hill surface."""
    sx = x_to_screen(x)
    sy = y_to_screen(hill_y(x))
    return sx, sy

def emit_dust(x_env, vel_x, amount=1):
    """Emit 'amount' dust puffs near car, drifting opposite to motion."""
    global dust_particles, _last_dust_spawn
    now = time.time()
    if now - _last_dust_spawn < DUST_SPAWN_COOLDOWN:
        return
    _last_dust_spawn = now

    sx, sy = env_to_screen_xy(x_env)
    direction = -1 if vel_x > 0 else (1 if vel_x < 0 else 0)

    for _ in range(amount):
        if len(dust_particles) >= DUST_MAX:
            break
        # Randomize spawn around wheels (slightly below car body)
        jitter_x = np.random.uniform(-10, 10)
        jitter_y = np.random.uniform(4, 10)
        px = sx + jitter_x
        py = sy - jitter_y

        # Initial velocity opposite to car’s movement + small upward drift
        vx = np.random.uniform(12, 20) * direction + np.random.uniform(-4, 4)
        vy = -np.random.uniform(8, 16)

        # Each particle: dict with position, velocity, life
        dust_particles.append({
            "x": px,
            "y": py,
            "vx": vx,
            "vy": vy,
            "born": now,
            "life": DUST_LIFETIME,
            "size": DUST_BASE_SIZE + np.random.uniform(-2, 2),
        })

def update_dust(dt):
    """Integrate particle motion and fade out / remove expired."""
    global dust_particles
    alive = []
    for p in dust_particles:
        age = time.time() - p["born"]
        if age > p["life"]:
            continue
        # simple physics: move + slight gravity + friction
        p["x"] += p["vx"] * dt
        p["y"] += p["vy"] * dt
        p["vy"] += 25 * dt          # gravity pulls down (screen coords)
        p["vx"] *= (1 - 0.6*dt)     # air drag
        p["vy"] *= (1 - 0.2*dt)

        alive.append(p)
    dust_particles = alive

def draw_dust(screen):
    """Draw soft circles with alpha that shrink over time."""
    for p in dust_particles:
        age = time.time() - p["born"]
        t = max(0.0, min(1.0, 1 - age / p["life"]))  # 1 → 0
        alpha = int(180 * (t ** 1.5))
        size = max(1, int(p["size"] * (0.6 + 0.4 * t)))
        # create a small surface with alpha
        s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*DUST_COLOR, alpha), (size, size), size)
        screen.blit(s, (p["x"] - size, p["y"] - size))


# -----------------------------
# GRAPHICS
# -----------------------------
clouds = [
    [100, 90], [350, 120], [700, 80]
]

# -------- Sounds --------
engine_sound = pygame.mixer.Sound("engine.wav")
success_sound = pygame.mixer.Sound("success.wav")
engine_sound.set_volume(0.35)   # adjust loudness
success_sound.set_volume(0.7)


def draw_cloud(screen, x, y):
    pygame.draw.ellipse(screen, (245,245,255), (x, y, 80, 40))
    pygame.draw.ellipse(screen, (245,245,255), (x+30, y-20, 80, 50))
    pygame.draw.ellipse(screen, (245,245,255), (x+60, y, 80, 40))

def update_clouds():
    for c in clouds:
        c[0] += 0.4
        if c[0] > WIDTH + 100:
            c[0] = -100

def draw_sky(screen):
    for i in range(HEIGHT):
        t = i / HEIGHT
        r = int(100*(1-t) + 160*t)
        g = int(160*(1-t) + 210*t)
        b = int(230*(1-t) + 255*t)
        pygame.draw.line(screen, (r,g,b), (0,i), (WIDTH,i))

    update_clouds()
    for c in clouds:
        draw_cloud(screen, c[0], c[1])

def draw_hills(screen):
    pts = []
    for sx in range(WIDTH):
        x = X_MIN + (sx - 50)*(X_MAX - X_MIN) / max(1, (WIDTH - 100))
        y = hill_y(x)
        pts.append((sx, y_to_screen(y)))

    pygame.draw.polygon(screen, (60,180,90), [(0,HEIGHT)] + pts + [(WIDTH,HEIGHT)])
    pygame.draw.lines(screen, (30,120,50), False, pts, 4)

def draw_flag(screen):
    gx = x_to_screen(GOAL_X)
    gy = y_to_screen(hill_y(GOAL_X))
    pygame.draw.line(screen, (230,230,230), (gx, gy-60), (gx, gy), 4)
    wave = math.sin(time.time()*4)*6
    pygame.draw.polygon(screen, (255,120,40),
        [(gx, gy-60), (gx+35, gy-55+wave), (gx, gy-40)]
    )

def draw_car(screen, pos_x, vel_x):
    x = pos_x
    y = hill_y(x)

    dx = 0.002
    dy = hill_y(x+dx) - hill_y(x-dx)
    angle = math.degrees(math.atan2(dy, 2*dx))

    cx = x_to_screen(x)
    cy = y_to_screen(y)

    body = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
    pygame.draw.rect(body, (250,80,80), (0, CAR_H//2, CAR_W, CAR_H//2), border_radius=6)
    pygame.draw.rect(body, (255,240,120), (6, 0, CAR_W-12, CAR_H//2), border_radius=6)
    body_rot = pygame.transform.rotate(body, -angle)
    rect = body_rot.get_rect(center=(cx, cy-8))
    screen.blit(body_rot, rect)

    spin = (pygame.time.get_ticks()/80) % 360
    wheel_surface = pygame.Surface((WHEEL_R*2, WHEEL_R*2), pygame.SRCALPHA)
    pygame.draw.circle(wheel_surface, (30,30,30), (WHEEL_R, WHEEL_R), WHEEL_R)
    pygame.draw.line(
        wheel_surface, (255,255,255), (WHEEL_R, WHEEL_R),
        (WHEEL_R+WHEEL_R*math.cos(spin), WHEEL_R+WHEEL_R*math.sin(spin)), 3
    )

    for wx in [-CAR_W//3, CAR_W//3]:
        wheel = pygame.transform.rotate(wheel_surface, -angle)
        wrect = wheel.get_rect(center=(cx+wx, cy-2))
        screen.blit(wheel, wrect)

def draw_hud(screen, step, ret, mode):
    font = pygame.font.SysFont("Arial", 18)
    txt = f"Step: {step}   Return: {ret:.1f}   Mode: {mode}"
    screen.blit(font.render(txt, True, (20,20,20)), (10,10))

# -----------------------------
# MAIN
# -----------------------------
def main():
    use_random = "--random" in sys.argv
    qpath = None
    for a in sys.argv[1:]:
        if a.startswith("--qpath="):
            qpath = a.split("=",1)[1]

    Q = None
    if qpath:
        print("Loading Q-table:", qpath)
        Q = np.load(qpath, allow_pickle=False)

    pygame.init()
    engine_sound.play(loops=-1)  # loop engine sound forever

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fancy MountainCar")
    clock = pygame.time.Clock()

    env = gym.make(ENV_ID)
    obs, _ = env.reset()

    mode = "Random" if use_random else ("Q-table" if Q is not None else "Keyboard")
    step = 0
    ep_ret = 0
    last_r_switch = time.time()

    running = True
    while running:
        action = 1  # no push
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        if Q is None and not use_random:
            if keys[pygame.K_LEFT]:  action = 0
            if keys[pygame.K_RIGHT]: action = 2
            if keys[pygame.K_r] and time.time() - last_r_switch > 0.3:
                use_random = not use_random
                mode = "Random" if use_random else "Keyboard"
                last_r_switch = time.time()

        if use_random:
            action = env.action_space.sample()
        elif Q is not None:
            pi, vi = discretize(obs)
            action = np.argmax(Q[pi, vi])

        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        step += 1
        ep_ret += reward

        # ---- DUST EMISSION (when moving / accelerating) ----
        pos_x, vel_x = obs[0], obs[1]
        if abs(vel_x) > DUST_SPAWN_SPEED and action != 1:  # only when pushing left/right
            # Emit more when faster
            emit_dust(pos_x, vel_x, amount=1 + int(abs(vel_x) * 30))


        # check if reached goal
        if obs[0] >= GOAL_X:
            success_sound.play()

        if done:
            dust_particles.clear()   # <--- New line
            obs, _ = env.reset()
            step = 0
            ep_ret = 0





        dt = clock.get_time() / 1000.0  # seconds since last frame
        update_dust(dt)

        draw_sky(screen)
        draw_hills(screen)
        draw_flag(screen)

        # draw dust behind the car
        draw_dust(screen)

        # draw the car on top
        draw_car(screen, obs[0], obs[1])
        draw_hud(screen, step, ep_ret, mode)




        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    engine_sound.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
