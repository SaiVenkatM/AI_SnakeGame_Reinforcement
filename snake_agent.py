
import numpy as np
import helper
import random
#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
    def helper_func(self, state):
        print("IN helper_func")
        snake_head_x = state[0]
        snake_head_y = state[1]
        snake_body = state[2]
        food_x = state[3]
        food_y = state[4]

        # snake_head_x
        if snake_head_x <= helper.BOARD_LIMIT_MIN:
            snake_head_x_prime = 1
        elif snake_head_x >= helper.BOARD_LIMIT_MAX:
            snake_head_x_prime = 2
        else:
            snake_head_x_prime = 0

        # snake_head_y
        if snake_head_y <= helper.BOARD_LIMIT_MIN:
            snake_head_y_prime = 1
        elif snake_head_y >= helper.BOARD_LIMIT_MAX:
            snake_head_y_prime = 2
        else:
            snake_head_y_prime = 0

        # (0,0) case
        if snake_head_x <= 0 or snake_head_y <= 0 or snake_head_x >= helper.IN_WALL_COORD or snake_head_y >= helper.IN_WALL_COORD:
            snake_head_x_prime, snake_head_y_prime = 0, 0

        # food_x
        if snake_head_x > food_x:
            food_x_prime = 1
        elif snake_head_x < food_x:
            food_x_prime = 2
        else:
            food_x_prime = 0

        # food_y
        if snake_head_y > food_y:
            food_y_prime = 1
        elif snake_head_y < food_y:
            food_y_prime = 2
        else:
            food_y_prime = 0

        # snake_body

        # body in top
        if (snake_head_x, snake_head_y - helper.BOARD_LIMIT_MIN) in snake_body:
            body_top = 1
        else:
            body_top = 0
        # body in bottom
        if (snake_head_x, snake_head_y + helper.BOARD_LIMIT_MIN) in snake_body:
            body_bottom = 1
        else:
            body_bottom = 0
        # body in left
        if (snake_head_x - helper.BOARD_LIMIT_MIN, snake_head_y) in snake_body:
            body_left = 1
        else:
            body_left = 0
        # body in right
        if (snake_head_x + helper.BOARD_LIMIT_MIN, snake_head_y) in snake_body:
            body_right = 1
        else:
            body_right = 0

        state_prime = (snake_head_x_prime, snake_head_y_prime, food_x_prime, food_y_prime, body_top, body_bottom, body_left, body_right)

        return state_prime


    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1
    def agent_action(self, state, points, dead):
    
        print("IN AGENT_ACTION")
        # Process the current state using a helper function
        processed_state = self.helper_func(state)

        # Get the current action
        current_action = self.a

        # Compute the reward based on the game state
        current_reward = self.compute_reward(points, dead)

        # Check if the agent is in training mode
        if self._train:

            # Update Q-values using the SARSA algorithm
            if self.s != None:
                next_max_q = np.max(self.Q[processed_state])
                alpha = self.LPC / (self.LPC + self.N[self.s + (current_action,)])
                current_q = self.Q[self.s + (current_action,)]
                self.Q[self.s + (current_action,)] += alpha * (current_reward + self.gamma * next_max_q - current_q)

            # Implement epsilon-greedy strategy for exploration
            max_q_value = float('-inf')
            selected_action = 0
            for i in range(3, -1, -1):
                if self.N[processed_state + (i,)] < self.Ne:
                    val = 1
                else:
                    val = self.Q[processed_state + (i,)]

                if val > max_q_value:
                    max_q_value = val
                    selected_action = i

            if not dead:
                self.N[processed_state + (selected_action,)] += 1
                self.points = points
            self.s = processed_state
            self.a = selected_action
        else:
            selected_action = np.argmax(self.Q[processed_state])

        # If the agent is dead, reset its state
        if dead:
            self.reset()

        # Return the selected action
        return selected_action
