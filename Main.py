import pygame
import time
import sys
import random
import copy
import numpy as np
from collections import deque


pygame.init()

# Constants
FPS = 60
BOARD_SIZE = [10, 10]
GAME_SCREEN_SIZE = [170, 170]
GAMES_PER_EVOLUTION = [4, 4]  # Adjusted for 3 x 2 set
SCREEN_SIZE = [GAME_SCREEN_SIZE[0] * GAMES_PER_EVOLUTION[0], GAME_SCREEN_SIZE[1] * GAMES_PER_EVOLUTION[1]]
SCREEN = pygame.display.set_mode((SCREEN_SIZE[0], SCREEN_SIZE[1]))
pygame.display.set_caption("Snake with AI")

class Board:

    def __init__(self, offsetx, offsety, q_learning):
        self.Grid = []
        self.offsetx = offsetx
        self.offsety = offsety
        self.snake = None
        self.running = True
        self.fruit = []
        self.num_fruit = 5
        self.font = pygame.font.Font(None, 20)
        self.q_learning = q_learning
        self.setup()
        self.state = self.get_state()

    def setup(self):
        self.time = 0
        self.snake = Snake([int(BOARD_SIZE[0]/2), int(BOARD_SIZE[1]/2)])
        for i in range(BOARD_SIZE[0]):
            self.Grid.append([0] * BOARD_SIZE[1])

        for _ in range(self.num_fruit):
            self.add_fruit()

    def get_state(self):
        snake_head = self.snake.body[0]
        fruit_positions = [fruit.pos for fruit in self.fruit]
        return (snake_head.x, snake_head.y, tuple(fruit_positions), self.snake.direction)

    def get_reward(self):
        reward = 0
        for fruit in self.fruit:
            if self.snake.body[0] == fruit.pos:
                reward = 10
        if self.collisions():
            reward = -100
        return reward

    def update_game(self):
        action = self.q_learning.get_action(self.state)
        self.snake.change_direction(action)
        next_state = self.get_state()
        reward = self.get_reward()
        done = self.collisions()  # Get the done flag from the collisions method
        self.q_learning.update_q_table(self.state, action, reward, next_state, done)
        self.state = next_state

    def update(self):
        self.Grid = [[0] * BOARD_SIZE[1] for _ in range(BOARD_SIZE[0])]
        for val in self.snake.body:
            try:
                self.Grid[int(val.x)][int(val.y)] = 1
            except IndexError:
                pass

    def add_fruit(self):
        confirmed = True
        while confirmed:
            pos = [random.randint(0, BOARD_SIZE[0]-1), random.randint(0, BOARD_SIZE[1]-1)]
            if pos not in self.snake.body:
                confirmed = False
        self.fruit.append(Fruit((pos[0], pos[1]), self.offsetx, self.offsety))

    def draw(self):
        rectsize = [GAME_SCREEN_SIZE[0] / BOARD_SIZE[0], GAME_SCREEN_SIZE[1] / BOARD_SIZE[1]]
        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                rect = pygame.Rect(i * rectsize[0] + self.offsetx, j * rectsize[1] + self.offsety, rectsize[0], rectsize[1])
                pygame.draw.rect(SCREEN, "black", rect, 1)

        for fruit in self.fruit:
            fruit.draw()

        border = pygame.Rect(self.offsetx - 2, self.offsety - 2, GAME_SCREEN_SIZE[0] + 4, GAME_SCREEN_SIZE[1] + 4)
        pygame.draw.rect(SCREEN, "black", border, 4)

        score = self.font.render("Score: " + str((self.snake.length * 10)-10), True, "blue")
        SCREEN.blit(score, (self.offsetx + 5, self.offsety + 5))

    def draw_snake(self):
        rectsize = [GAME_SCREEN_SIZE[0] / BOARD_SIZE[0], GAME_SCREEN_SIZE[1] / BOARD_SIZE[1]]
        for segment in self.snake.body:
            segment_rect = pygame.Rect(segment[0] * rectsize[0] + 1 + self.offsetx, segment[1] * rectsize[1] + 1 + self.offsety, rectsize[0] - 2, rectsize[1] - 2)
            pygame.draw.rect(SCREEN, "green", segment_rect)

    def collisions(self):
        for fruit in self.fruit:
            if fruit.pos == self.snake.body[0]:
                self.snake.add_body()
                self.fruit.remove(fruit)
                self.add_fruit()
            else:
                for pos in self.snake.body[1:]:
                    if pos == self.snake.body[0] and self.snake.length > 2:
                        return True
        if self.snake.body[0].x < 0 or self.snake.body[0].x > BOARD_SIZE[0] - 1:
            return True
        elif self.snake.body[0].y < 0 or self.snake.body[0].y > BOARD_SIZE[1] - 1:
            return True
        else:
            return False

    def run(self):

        self.draw()
        
        if self.collisions() and self.running:
            self.running = False
            print("Fitness: ", self.snake.length * 10 + self.time/5)
            return self.snake.length * 10 + self.time/5
        else:
            self.time += 1
        
        self.update_game()
        if self.running:
            self.draw_snake()
        self.snake.run()


class Snake:

    def __init__(self, pos):
        self.body = [pygame.math.Vector2(pos[0], pos[1])]
        self.length = 1
        self.direction = {"Right": True, "Left": False, "Up": False, "Down": False}
        self.movecooldown = 0

    def change_direction(self, newdirection):
        if self.length > 1:
            if (newdirection == "Right" and self.direction["Left"]) or \
               (newdirection == "Left" and self.direction["Right"]) or \
               (newdirection == "Up" and self.direction["Down"]) or \
               (newdirection == "Down" and self.direction["Up"]):
                return
        for direction, value in self.direction.items():
            if value:
                self.direction[direction] = False
        self.direction[newdirection] = True

    def add_body(self):
        tail_position = self.body[-1] if len(self.body) > 1 else self.body[0]
        self.body.append(pygame.math.Vector2(tail_position.x, tail_position.y))
        self.length += 1

    def move(self):
        if self.direction["Right"]:
            new_head = pygame.math.Vector2(self.body[0].x + 1, self.body[0].y)
        elif self.direction["Left"]:
            new_head = pygame.math.Vector2(self.body[0].x - 1, self.body[0].y)
        elif self.direction["Down"]:
            new_head = pygame.math.Vector2(self.body[0].x, self.body[0].y + 1)
        elif self.direction["Up"]:
            new_head = pygame.math.Vector2(self.body[0].x, self.body[0].y - 1)
        self.body.insert(0, new_head)
        if len(self.body) > self.length:
            self.body.pop()

    def run(self):
        if time.time() - self.movecooldown > 0.05:
            self.movecooldown = time.time()
            self.move()


class Fruit:

    def __init__(self, pos, offsetx, offsety):
        self.pos = pygame.math.Vector2(pos[0], pos[1])
        self.offsetx = offsetx
        self.offsety = offsety

    def draw(self):
        rectsize = [GAME_SCREEN_SIZE[0] / BOARD_SIZE[0], GAME_SCREEN_SIZE[1] / BOARD_SIZE[1]]
        fruit = pygame.Rect(self.pos.x * rectsize[0] + 1 + self.offsetx, self.pos.y * rectsize[1] + 1 + self.offsety, rectsize[0] - 2, rectsize[1] - 2)
        pygame.draw.rect(SCREEN, "red", fruit)


class QLearning:
    def __init__(self, learning_rate, discount_factor, exploration_rate, buffer_size):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.replay_buffer = deque(maxlen=buffer_size)

    def get_state_key(self, state):
        snake_head_x, snake_head_y, fruit_positions, direction_dict = state
        fruit_positions_tuple = tuple((pos.x, pos.y) for pos in fruit_positions)
        direction_tuple = (direction_dict['Right'], direction_dict['Left'], direction_dict['Up'], direction_dict['Down'])
        return (snake_head_x, snake_head_y, fruit_positions_tuple, direction_tuple)

    def get_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.exploration_rate:
            return random.choice(["Up", "Down", "Left", "Right"])
        else:
            if state_key in self.q_table:
                return max(self.q_table[state_key], key=self.q_table[state_key].get)
            else:
                return random.choice(["Up", "Down", "Left", "Right"])

    def update_q_table(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {"Up": 0, "Down": 0, "Left": 0, "Right": 0}

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {"Up": 0, "Down": 0, "Left": 0, "Right": 0}

        # Double Q-Learning
        next_action = max(self.q_table[next_state_key], key=self.q_table[next_state_key].get)
        next_q_value = self.q_table[next_state_key][next_action]

        target_q_value = reward + (self.discount_factor * next_q_value * (not done))
        td_error = target_q_value - self.q_table[state_key][action]

        self.q_table[state_key][action] += self.learning_rate * td_error

        # Experience Replay
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) >= self.replay_buffer.maxlen:
            self.learn_from_replay()

    def learn_from_replay(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)

        for state, action, reward, next_state, done in batch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)

            if state_key not in self.q_table:
                self.q_table[state_key] = {"Up": 0, "Down": 0, "Left": 0, "Right": 0}

            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {"Up": 0, "Down": 0, "Left": 0, "Right": 0}

            # Double Q-Learning
            next_action = max(self.q_table[next_state_key], key=self.q_table[next_state_key].get)
            next_q_value = self.q_table[next_state_key][next_action]

            target_q_value = reward + (self.discount_factor * next_q_value * (not done))
            td_error = target_q_value - self.q_table[state_key][action]

            self.q_table[state_key][action] += self.learning_rate * td_error


def reset(individuals):
    for i in range(len(individuals)):
        i_offset = i // GAMES_PER_EVOLUTION[1]
        j_offset = i % GAMES_PER_EVOLUTION[1]
        x_offset = j_offset * GAME_SCREEN_SIZE[0]
        y_offset = i_offset * GAME_SCREEN_SIZE[1]
        individuals[i] = Board(x_offset, y_offset, individuals[i].q_learning)


def display_generations(generation):
    font = pygame.font.Font(None, 36)
    text = font.render("Generation: " + str(generation), True, (0, 0, 0))
    SCREEN.blit(text, (20, 20))


def mutate(individual, mutation_rate):
    mutated_q_learning = copy.deepcopy(individual.q_learning)
    for state_key, actions in mutated_q_learning.q_table.items():
        for action, value in actions.items():
            if random.random() < mutation_rate:
                mutated_q_learning.q_table[state_key][action] = random.uniform(-1, 1)
    return mutated_q_learning

def crossover(parent1, parent2):
    child_q_learning = copy.deepcopy(parent1.q_learning)

    # Perform crossover between parent q-tables
    for state_key, parent1_actions in parent1.q_learning.q_table.items():
        if state_key in parent2.q_learning.q_table:
            parent2_actions = parent2.q_learning.q_table[state_key]
            child_actions = child_q_learning.q_table[state_key]
            for action in parent1_actions:
                if random.random() < 0.5:
                    child_actions[action] = parent1_actions[action]
                else:
                    child_actions[action] = parent2_actions[action]
        else:
            # If state_key is missing in parent2, choose actions from parent1
            child_q_learning.q_table[state_key] = parent1_actions.copy()

    return child_q_learning


def evolve(individuals, mutation_rate):
    individuals.sort(key=lambda x: x.snake.length, reverse=True)
    best_individuals = individuals[:2]  # Select the two fittest snakes
    new_individuals = []

    while len(new_individuals) < len(individuals):
        # Breed the two fittest snakes
        child_q_learning = crossover(best_individuals[0], best_individuals[1])
        new_individual = Board(0, 0, child_q_learning)
        new_individuals.append(new_individual)

    # Mutate all individuals
    for individual in new_individuals:
        mutated_q_learning = mutate(individual, mutation_rate)
        individual.q_learning = mutated_q_learning

    return new_individuals



def main():
    individuals = []
    num = 0
    for i in range(0, GAMES_PER_EVOLUTION[0]):
        for j in range(0, GAMES_PER_EVOLUTION[1]):
            x_offset = i * GAME_SCREEN_SIZE[0]
            y_offset = j * GAME_SCREEN_SIZE[1]
            q_learning = QLearning(learning_rate=0.3, discount_factor=0.8, exploration_rate=0.6, buffer_size = 10000)
            game = Board(x_offset, y_offset, q_learning)
            individuals.append(game)
            num += 1

    running = True
    clock = pygame.time.Clock()
    generation = 1
    fitness = []
    while running:
        SCREEN.fill("white")
        all_dead = True
        for individual in individuals:
            length = individual.run()
            if length is not None:
                fitness.append(length)

        if len(fitness) == len(individuals):
            individuals = evolve(individuals, mutation_rate=0.4)
            reset(individuals)
            print("Generation:", generation, "Evolved:", len(individuals), "individuals")
            generation += 1
            fitness = []

        display_generations(generation)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                sys.exit()

        clock.tick(FPS)

if __name__ == "__main__":
    main()
