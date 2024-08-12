import numpy as np
from scipy.integrate import solve_ivp

import numpy as np
from scipy.integrate import solve_ivp

import numpy as np
from scipy.integrate import solve_ivp

class Cart:
    def __init__(self, mass, friction_coeff, spring_constant, spring_max, wall_left, wall_right, cart_width, pendulum_length, pendulum_friction, gravity=9.81):
        self.mass = mass
        self.friction_coeff = friction_coeff
        self.spring_constant = spring_constant
        self.spring_max = spring_max
        self.wall_left = wall_left
        self.wall_right = wall_right
        self.cart_width = cart_width
        self.pendulum_length = pendulum_length
        self.gravity = gravity
        self.pendulum_friction = pendulum_friction

        # Initial state: [cart_position, cart_velocity, pendulum_angle (theta), pendulum_angular_velocity (theta_dot)]
        self.state = np.array([(self.wall_right - self.wall_left)/2, 0.0, np.pi/4, 0.0])  # Start with pendulum at 45 degrees

    def dynamics(self, t, state, user_force):
        x, v, theta, theta_dot = state

        # Friction force
        friction_force = self.friction_coeff * v

        # Spring forces
        left_spring_force = self.spring_constant * (self.wall_left + self.spring_max - (x - self.cart_width / 2)) \
            if x - self.cart_width / 2 < self.wall_left + self.spring_max else 0
        
        right_spring_force = self.spring_constant * ((x + self.cart_width / 2) - (self.wall_right - self.spring_max)) \
            if x + self.cart_width / 2 > self.wall_right - self.spring_max else 0

        # Net force
        net_force = user_force - friction_force + left_spring_force - right_spring_force

        # Cart acceleration
        acceleration = net_force / self.mass

        # Pendulum dynamics
        theta_ddot = -(acceleration / self.pendulum_length) * np.cos(theta) - (self.gravity / self.pendulum_length) * np.sin(theta) - theta_dot*self.pendulum_friction

        return [v, acceleration, theta_dot, theta_ddot]

    def step(self, user_force, dt):
        # Use solve_ivp with RK45 (an adaptive Runge-Kutta method)
        sol = solve_ivp(self.dynamics, [0, dt], self.state, args=(user_force,), method='RK45', t_eval=[dt])
        self.state = sol.y[:, -1]

    def get_position(self):
        return self.state[0]

    def get_velocity(self):
        return self.state[1]

    def get_pendulum_angle(self):
        return self.state[2]

    def get_pendulum_angular_velocity(self):
        return self.state[3]

    def reset(self, position=0.0, velocity=0.0, theta=np.pi/4, theta_dot=0.0):
        self.state = np.array([position, velocity, theta, theta_dot])

def simulate_cart(cart, duration, dt, user_force_func):
    t = 0.0
    positions = []
    velocities = []
    angles = []
    angular_velocities = []

    while t < duration:
        user_force = user_force_func(t)
        cart.step(user_force, dt)
        positions.append(cart.get_position())
        velocities.append(cart.get_velocity())
        angles.append(cart.get_pendulum_angle())
        angular_velocities.append(cart.get_pendulum_angular_velocity())
        t += dt

    return np.array(positions), np.array(velocities), np.array(angles), np.array(angular_velocities)







def main():
    # Set up the cart with physical parameters
    cart = Cart(
        mass=1.0,
        friction_coeff=2.0,  # b
        spring_constant=100.0,  # k
        spring_max=100.0,
        wall_left=0.0,
        wall_right=800.0,
        cart_width=60.0,
        pendulum_length=2.0  # Length of the pendulum
    )

    # Define a user force function (for example, a sinusoidal force over time)
    def user_force_func(t):
        return 10.0 * np.sin(t)

    # Simulate the cart for 10 seconds with a time step of 0.01 seconds
    positions, velocities = simulate_cart(cart, duration=10.0, dt=0.01, user_force_func=user_force_func)

    # The positions and velocities arrays now contain the simulation results
    print(positions)


if __name__ == "main":
    main()