
"""
Build a simple simulation of a series of agents with attractive and repulsive forces moving through a space. This can serve as a desktop background or LinkedIn profile pic maybe...
"""

# ================================================================
# Import libraries

# Numerical libraries
import numpy as np
import random
from scipy.spatial import KDTree

# Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import matplotlib.collections as mcoll

# Presentation libraries
from screeninfo import get_monitors
monitor = get_monitors()[0]

# Global parameters

DT = 0.05 # Update every 0.05s
SIM_RUN = 600 # Run for 10 mins before resetting
# SCREEN_WIDTH = monitor.width
# SCREEN_HEIGHT = monitor.height
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Variable inputs
NUM_AGENTS = 100
AGENT_TYPES = 3
REPEL = 5000
ATTRACT = 5000 # These will be overriden by slider on GUI
INTERACTION_RADIUS = 200
AGENT_RADIUS = 5
MAX_SPEED = 150
MIN_SPEED = 50

# Colour scheme
background = (0.1, 0.1, 0.1) # Dark grey
colour_map = plt.get_cmap('viridis') # Colour map for agents
# agent_colours = [colour_map(i) for i in np.linspace(0, 1, AGENT_TYPES)]
agent_colours = [(0.2, 0.4, 1.0), (0.8, 0.2, 1.0), (1.0, 0.2, 0.2)] # Blue, Purple, Red

# ================================================================
# Define a class for each of the agent, that takes it's type, position, direction and speed at any given moment

class Agent():
    def __init__(self, type, pos, vel, repel, attract):

        self.type = type # 1, 2, 3 (Blue, Purple, Red)
        self.position = pos # x, y positions
        self.velocity = vel # x, y speeds
        self.interaction = {} # Store the closest neighour ID with interaction time - continually update for each agent with timestep dt
        self.repulsive_force = repel # Easy manipulation of the strength of attractive/repulsive forces between agents
        self.attractive_force = attract
        self.trail = [pos.copy()] # Store previous positions for trail effect

    def move(self, dt): # Take as input the change in time
        self.position += self.velocity * dt
        self.trail.append(self.position.copy())
        if len(self.trail) > 20:  # Limit trail length
            self.trail.pop(0)

# ================================================================
# Define a series of rules governing the interactions between agents

def interact(agent1, agent2, dt): # Agent 1 reacts to agent 2 always

    """
    (1) Blue agents strongly repel other blues, attract reds and have no influcen on purples
    (2) Red agents strongly repel other reds, attract purples and have no influence on blues
    (3) Purple agents strongly repel other purples, attract blues and have no influence on reds

    Agents will interact with their closest neighbour, accelerating for as long as they are neighbours.
    """

    key = id(agent2)
    agent1.interaction[key] = agent1.interaction.get(key, 0) + dt # Increment timestep
    direction = agent2.position - agent1.position # If this value is negative, they are trevlling away from one another already
    distance = np.linalg.norm(direction)
    if distance == 0 or distance > INTERACTION_RADIUS: # Prevent division by zero
        return
    direction = direction / distance # Normalise the direction vector

    # Repel own type
    if agent1.type == agent2.type:
        force = -agent1.repulsive_force * direction / distance**2
    # Attract next type (cyclic)
    elif agent2.type == (agent1.type % AGENT_TYPES) + 1:
        force = agent1.attractive_force * direction / distance**2
    # No influence
    else:
        force = np.array([0.0, 0.0])

    # Update velocity based on force and interaction time
    agent1.velocity += force * agent1.interaction[key] * dt
    speed = np.linalg.norm(agent1.velocity)
    if speed > MAX_SPEED: # Cap max speed
        agent1.velocity = (agent1.velocity / speed) * MAX_SPEED
    elif speed < MIN_SPEED: # Cap min speed
        agent1.velocity = (agent1.velocity / speed) * MIN_SPEED

# Reset function - generate starting positions and velocities for each agent
def reset_agents(num_agents, agent_types, screen_width, screen_height, repel, attract):

    agents = []
    for i in range(num_agents):
        type = random.randint(1, agent_types) # Randomly assign type
        pos = np.array([random.uniform(0, screen_width), random.uniform(0, screen_height)]) # Random starting position
        angle = random.uniform(0, 2 * np.pi) # Random starting direction
        speed = random.uniform(MIN_SPEED, MAX_SPEED) # Random starting speed
        vel = np.array([speed * np.cos(angle), speed * np.sin(angle)]) # Convert to x,y components
        agents.append(Agent(type, pos, vel, repel, attract))
    
    return agents # Store and return an array of the agents created - each agent is a class with its own properties and methods

# ================================================================
# Set up the figure and axis for the animation

def setup_figure(screen_width, screen_height):

    fig, ax = plt.subplots()
    fig.set_size_inches(screen_width / 100, screen_height / 100) # Set figure size to screen size (assuming 100 dpi)
    fig.patch.set_facecolor(background) # Set figure background color
    ax.set_xlim(0, screen_width)
    ax.set_ylim(0, screen_height)
    ax.set_facecolor(background)
    ax.axis('off') # Turn off axes
    plt.tight_layout()
    return fig, ax

# ================================================================
# Main function to run the simulation

def run_simulation():
    global ATTRACT, REPEL

    agents = reset_agents(NUM_AGENTS, AGENT_TYPES, SCREEN_WIDTH, SCREEN_HEIGHT, REPEL, ATTRACT)
    fig, ax = setup_figure(SCREEN_WIDTH, SCREEN_HEIGHT)

    scatter = ax.scatter([agent.position[0] for agent in agents],
                         [agent.position[1] for agent in agents],
                         c=[agent_colours[agent.type - 1] for agent in agents],
                         s=AGENT_RADIUS**2)

    trails = []
    for agent in agents:
        lc = mcoll.LineCollection([], colors=[agent_colours[agent.type - 1]], linewidths=2)
        ax.add_collection(lc)
        trails.append(lc)

    def update(frame):
        nonlocal agents
        dt = DT

        for agent in agents:
            agent.attractive_force = ATTRACT
            agent.repulsive_force = REPEL

        # Build KDTree for agent positions
        positions = np.array([agent.position for agent in agents])
        tree = KDTree(positions)

        # Efficient neighbor search
        for i, agent1 in enumerate(agents):
            idxs = tree.query_ball_point(agent1.position, INTERACTION_RADIUS)
            for j in idxs:
                if i != j:
                    interact(agent1, agents[j], dt)
            agent1.move(dt)

            # Boundary conditions
            if agent1.position[0] < AGENT_RADIUS:
                agent1.position[0] = AGENT_RADIUS
                agent1.velocity[0] *= -1
            elif agent1.position[0] > SCREEN_WIDTH - AGENT_RADIUS:
                agent1.position[0] = SCREEN_WIDTH - AGENT_RADIUS
                agent1.velocity[0] *= -1

            if agent1.position[1] < AGENT_RADIUS:
                agent1.position[1] = AGENT_RADIUS
                agent1.velocity[1] *= -1
            elif agent1.position[1] > SCREEN_HEIGHT - AGENT_RADIUS:
                agent1.position[1] = SCREEN_HEIGHT - AGENT_RADIUS
                agent1.velocity[1] *= -1

        # Ball-ball collision detection and response
        # for i, agent1 in enumerate(agents):
        #     for j, agent2 in enumerate(agents):
        #         if i < j:
        #             delta = agent2.position - agent1.position
        #             dist = np.linalg.norm(delta)
        #             if dist < 2 * AGENT_RADIUS and dist > 0:
        #                 agent1.velocity, agent2.velocity = agent2.velocity.copy(), agent1.velocity.copy()
        #                 overlap = 2 * AGENT_RADIUS - dist
        #                 direction = delta / dist
        #                 agent1.position -= direction * overlap / 2
        #                 agent2.position += direction * overlap / 2

        scatter.set_offsets([agent.position for agent in agents])

        # Update trails with fading effect
        for trail, agent in zip(trails, agents):
            if len(agent.trail) > 1:
                segments = [[agent.trail[i], agent.trail[i+1]] for i in range(len(agent.trail)-1)]
                alphas = np.linspace(0.1, 0.8, len(segments))
                colors = [(*agent_colours[agent.type - 1][:3], alpha) for alpha in alphas]
                trail.set_segments(segments)
                trail.set_color(colors)
            else:
                trail.set_segments([])

        return (scatter, *trails)

    ani = animation.FuncAnimation(fig, update, frames=int(SIM_RUN / DT), interval=DT * 1000, blit=True)
    plt.show()

if __name__ == "__main__":

    run_simulation()

# ================================================================

"""
To do:
(1) Add sliders to control attractive and repulsive forces
(2) Add option to save as mp4 or gif
(3) Experiment with different force laws (e.g. linear, exponential decay)
(4) Add option for different boundary conditions (e.g. bounce off edges)
(5) Optimize performance for larger numbers of agents
(6) Add trails to agents to visualize paths
(7) Experiment with different color maps and sizes for agents
(8) Add option to pause and resume simulation
(9) Implement a GUI for more interactive control
(10) Explore 3D version of the simulation
"""