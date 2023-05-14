from santorinai.player import Player
from santorinai.tester import Tester


class MyPlayer(Player):
    """
    My player description
    """

    def name(self):
        return "My player name"

    # Placement of the pawns
    def place_pawn(self, board, pawn):
        my_pawn_number = pawn.number # Between 1 and 6 depending on the game mode
        my_player_number = pawn.player_number # Between 1 and 3 depending on the game mode

        # Do some magic here to choose a position
        my_choice = (2, 3) # A position on the 5x5 board

        return my_choice

    # Movement and building
    def play_move(self, board, pawn):
        my_initial_position = pawn.pos

        board_array = board.board # A 5x5 array of integers representing the board
        # 0: empty
        # 1: tower level 1
        # 2: tower level 2
        # 3: tower level 3
        # 4: terminated tower

        # Do some magic here to choose a position
        my_move_vector = (1, 1) # Moving top right
        my_build_vector = (1, 0) # Building right (relative to the new position)

        return my_move_vector, my_build_vector

p1 = MyPlayer()
p2 = MyPlayer()

tester = Tester()
tester.delay_between_moves = 1
tester.play_1v1(p1, p2)