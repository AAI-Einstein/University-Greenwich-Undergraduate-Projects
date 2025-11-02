import pygame
import numpy as np
import random
import math
import sys

# ------------------- PYGAME SETUP -------------------
pygame.init()
WIDTH, HEIGHT = 900, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DFO - Equilateral Triangles Following Flies")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Colors
BACKGROUND_COLOR = (25, 25, 25)
TEXT_COLOR = (255, 255, 255)
MOUSE_COLOR = (255, 255, 0)

# DFO visual constants
TRIANGLE_SIZE = 20
MOVE_SPEED = 0.05  # smooth interpolation speed
FPS = 60

# ------------------- DFO PARAMETERS -------------------
N = 20          # flies
D = 2           # dimensions
lowerB = np.array([-1.0, -1.0])
upperB = np.array([1.0, 1.0])
delta = 0.001
update_every = 50  # iterations between DFO updates

# Initialize flies
X = np.random.uniform(lowerB, upperB, (N, D))
fitness = np.zeros(N)

# ------------------- FUNCTIONS -------------------
def mouse_sphere(x, mouse_world):
    """Sphere-like fitness centered on mouse position."""
    return np.sum((x - mouse_world) ** 2)

def map_to_screen(x):
    """Convert DFO coordinates to screen coordinates."""
    sx = int((x[0] - lowerB[0]) / (upperB[0] - lowerB[0]) * WIDTH)
    sy = int((x[1] - lowerB[1]) / (upperB[1] - lowerB[1]) * HEIGHT)
    return sx, HEIGHT - sy

def map_from_screen(mouse_x, mouse_y):
    """Convert screen coords to DFO coords."""
    wx = lowerB[0] + (mouse_x / WIDTH) * (upperB[0] - lowerB[0])
    wy = lowerB[1] + ((HEIGHT - mouse_y) / HEIGHT) * (upperB[1] - lowerB[1])
    return np.array([wx, wy])

def draw_equilateral_triangle(center, angle, color):
    """Draw an equilateral triangle pointing toward a given angle."""
    x, y = center
    points = []
    for i in range(3):
        theta = angle + i * (2 * math.pi / 3)
        px = x + TRIANGLE_SIZE * math.cos(theta)
        py = y + TRIANGLE_SIZE * math.sin(theta)
        points.append((px, py))
    pygame.draw.polygon(screen, color, points)

# ------------------- INITIAL TRIANGLE VISUAL SETUP -------------------
# Pick 4 flies to visualize
s = 0
picks = [X[s, :]] + random.sample(list(X), 3)
colors = [(255, 0, 0), (0, 255, 0), (255, 105, 180), (0, 128, 255)]

# Start triangles at random screen positions
tri_positions = [np.random.rand(2) * np.array([WIDTH, HEIGHT]) for _ in range(4)]

# ------------------- MAIN LOOP -------------------
running = True
iteration = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_world = map_from_screen(mouse_x, mouse_y)

    # Every 100 iterations, run DFO to find new positions
    if iteration % update_every == 0:
        for i in range(N):
            fitness[i] = mouse_sphere(X[i, :], mouse_world)
        s = np.argmin(fitness)

        for i in range(N):
            if i == s:
                continue
            left = (i - 1) % N
            right = (i + 1) % N
            bNeighbour = right if fitness[right] < fitness[left] else left
            for d in range(D):
                if np.random.rand() < delta:
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])
                    continue
                u = np.random.rand()
                X[i, d] = X[bNeighbour, d] + u * (X[s, d] - X[i, d])
                if X[i, d] < lowerB[d] or X[i, d] > upperB[d]:
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])

        # Select 4 flies again (best + 3 random)
        picks = [X[s, :]] + random.sample(list(X), 3)

    # --- Smoothly move each triangle toward its assigned fly’s screen location ---
    screen.fill(BACKGROUND_COLOR)

    # Draw yellow circle (mouse target)
    pygame.draw.circle(screen, MOUSE_COLOR, (mouse_x, mouse_y), 10)

    for i, fly in enumerate(picks):
        target_screen = np.array(map_to_screen(fly), dtype=float)
        # move smoothly
        tri_positions[i] += (target_screen - tri_positions[i]) * MOVE_SPEED

        # Calculate direction for triangle tip
        dx, dy = target_screen - tri_positions[i]
        angle = math.atan2(dy, dx)
        draw_equilateral_triangle(tri_positions[i], angle, colors[i])

    # Display iteration text
    text = font.render(f"Iteration: {iteration} | Best fly fitness: {fitness[s]:.4f}", True, TEXT_COLOR)
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)
    iteration += 1

pygame.quit()
sys.exit()
