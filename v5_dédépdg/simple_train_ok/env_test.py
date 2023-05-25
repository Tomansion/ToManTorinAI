import numpy as np

class Env:
    def __init__(self):
        self.board_size = 5
        self.pawn_position = [0, 0]
        self.goal_position = [4, 4]
        self.num_states = self.board_size * self.board_size
        self.num_actions = 4
        self.last_action = None
        self.move_count = [0] * self.num_actions
        self.nb_win = 0

    def reset(self):
        self.pawn_position = [0, 0]
        return self._get_state()

    def step(self, action):
        if action == 0:  # Move up
            self.pawn_position[0] = max(0, self.pawn_position[0] - 1)
        elif action == 1:  # Move down
            self.pawn_position[0] = min(self.board_size - 1, self.pawn_position[0] + 1)
        elif action == 2:  # Move left
            self.pawn_position[1] = max(0, self.pawn_position[1] - 1)
        elif action == 3:  # Move right
            self.pawn_position[1] = min(self.board_size - 1, self.pawn_position[1] + 1)
        self.move_count[action] += 1

        done = self.pawn_position == self.goal_position
        if done:
            reward = 1
            self.nb_win += 1
        else:
            reward = 0 
        return self._get_state(), reward, done

    def _get_state(self):
        state = np.zeros((self.board_size, self.board_size))
        state[self.pawn_position[0], self.pawn_position[1]] = 1
        return state.flatten()
    
    def display(self):
        print("Move count: {}".format(self.move_count))
        print("Number of wins: {}".format(self.nb_win))
        print("Board:")
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.pawn_position == [i, j]:
                    print("P", end=" ")
                elif self.goal_position == [i, j]:
                    print("G", end=" ")
                else:
                    print(".", end=" ")
            print("")
        print("")