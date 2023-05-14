# ToManTorinAI

An AI that can play the Santoroni board game.

To participtate in [the SantorinAI project](https://github.com/Tomansion/SantorinAI).

## Principles

This AI uses reinforcement learning to learn how to play the game.

The ToManTorinAI will be presented with a board state and a set of possible actions. It will then choose an action and receive a reward based on the outcome of the action.

## V1 - DeathCursor

First version of the AI. One model to move and build, input spreaded as much as possible.

Algorithm used : Q value learning

### Inputs

- From 0 to 24 : 1 if tile is empty, 0 otherwise
- From 25 to 49 : 1 if towel level 1
- From 50 to 74 : 1 if towel level 2
- From 75 to 99 : 1 if towel level 3
- From 100 to 124 : 1 if towel terminated
- From 125 to 149 : 1 if playing pawn
- From 150 to 174 : 1 if ally pawn
- From 175 to 199 : 1 if enemy pawn 1
- From 200 to 224 : 1 if enemy pawn 2

### Outputs

Movements and build vectors:

| Vec   |      |      |     |     | Id  |     |     |
| ----- | ---- | ---- | --- | --- | --- | --- | --- |
| -1 1  | 0 1  | 1 1  |     |     | 0   | 1   | 2   |
| -1 0  | ---  | 1 0  |     |     | 7   | --- | 3   |
| -1 -1 | 0 -1 | 1 -1 |     |     | 6   | 5   | 4   |

- From 0 to 7 : highest output level to move on
- From 8 to 15 : highest output level to build on





