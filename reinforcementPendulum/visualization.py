import pygame
import sys
import math
from cart_object import Cart

# Initialize Pygame
pygame.init()

# Screen dimensions (in pixels)
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pendulum Simulation")

# Scale factor (pixels per meter)
scale_factor = WIDTH / 1.0  # 1 meter corresponds to the full width of the screen

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255, 128)  # Blue with some transparency

# Frame rate
clock = pygame.time.Clock()
FPS = 60

# Rail properties
rail_y = HEIGHT // 2

# Cart properties (in meters)
cart_width_m = 0.1  # 10 cm
cart_height_m = 0.05  # 5 cm

# Pendulum properties (in meters)
pendulum_length_m = 0.4  # 30 cm
bob_radius_m = 0.02  # 2 cm

# Breaking zone properties (in meters)
zone_width_m = 0.1  # 10 cm
zone_height_m = 0.15  # 15 cm

# Define max force (in Newtons)
max_force = 9  # Max force that can be applied to the cart

# Initialize the Cart object
cart = Cart(
    mass=1.0,  # in kg
    friction_coeff=4.0,  # b, in N·s/m
    spring_constant=100.0,  # k, in N/m
    spring_max=0.1,  # max spring compression, in meters
    wall_left=0.0,  # left wall position, in meters
    wall_right=1.0,  # right wall position, in meters (1 meter total track length)
    cart_width=cart_width_m,  # cart width in meters
    pendulum_length=pendulum_length_m,  # pendulum length in meters
    pendulum_friction=0.1
)

def draw_rail():
    pygame.draw.line(screen, BLACK, (0, rail_y), (WIDTH, rail_y), 5)

def draw_cart(x_m):
    x_pixels = x_m * scale_factor
    cart_width_pixels = cart_width_m * scale_factor
    cart_height_pixels = cart_height_m * scale_factor
    pygame.draw.rect(screen, RED, (x_pixels - cart_width_pixels // 2, rail_y - cart_height_pixels // 2, cart_width_pixels, cart_height_pixels))

def draw_pendulum(x_m, theta):
    x_pixels = x_m * scale_factor
    pendulum_length_pixels = pendulum_length_m * scale_factor
    bob_radius_pixels = bob_radius_m * scale_factor

    # Calculate pendulum bob position
    bob_x = x_pixels + pendulum_length_pixels * math.sin(theta)
    bob_y = rail_y + pendulum_length_pixels * math.cos(theta)

    # Draw the string
    pygame.draw.line(screen, BLACK, (x_pixels, rail_y), (bob_x, bob_y), 2)

    # Draw the pendulum bob
    pygame.draw.circle(screen, BLACK, (int(bob_x), int(bob_y)), int(bob_radius_pixels))

def draw_breaking_zones():
    zone_width_pixels = zone_width_m * scale_factor
    zone_height_pixels = zone_height_m * scale_factor

    # Left breaking zone
    left_zone = pygame.Surface((zone_width_pixels, zone_height_pixels))
    left_zone.set_alpha(128)
    left_zone.fill(BLUE)
    screen.blit(left_zone, (0, rail_y - zone_height_pixels // 2))

    # Right breaking zone
    right_zone = pygame.Surface((zone_width_pixels, zone_height_pixels))
    right_zone.set_alpha(128)
    right_zone.fill(BLUE)
    screen.blit(right_zone, (WIDTH - zone_width_pixels, rail_y - zone_height_pixels // 2))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get mouse x-axis position (since the joystick acts as a mouse)
    mouse_x, _ = pygame.mouse.get_pos()

    # Calculate force based on mouse x-axis position
    force = max_force * ((mouse_x / WIDTH) - 0.5) * 2  # Normalize to [-1, 1] and scale

    # Update the cart dynamics using the force and time step
    cart.step(force, 1 / FPS)

    # Get the updated cart position and pendulum angle
    cart_x_m = cart.get_position()
    theta = cart.get_pendulum_angle()

    # Clear the screen
    screen.fill(WHITE)

    # Draw the rail
    draw_rail()

    # Draw the breaking zones
    draw_breaking_zones()

    # Draw the cart
    draw_cart(cart_x_m)

    # Draw the pendulum
    draw_pendulum(cart_x_m, theta)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(FPS)
