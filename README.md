# ToManTorinAI

An AI that can play the Santoroni board game.

To participtate in [the SantorinAI project](https://github.com/Tomansion/SantorinAI).

## Principles

This AI uses reinforcement learning to learn how to play the game.

The ToManTorinAI will be presented with a board state and a set of possible actions. It will then choose an action and receive a reward based on the outcome of the action.

## V3 - DDPoG

Same as V2 but with a different algorithm.

Algorithm used : [DDPG](https://keras.io/examples/rl/ddpg_pendulum/)

## V2 - DeepInTheQ

Second version of the AI. One model to move and build, input spreaded as much as possible.

Algorithm used : [DQN](https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb)

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


### Results

Not better, fails a lot:

```md
# ====================================== After ~80 episodes
Player DeepInTheQ won 0 times (0.0%)
Player Firsty First won 100 times (100.0%)
We missplyed 1300 times over 2300 turns, (56.52173913043478%)

Player Firsty First won 100 times (100.0%)
Player DeepInTheQ won 0 times (0.0%)
We missplyed 200 times over 1000 turns, (20.0%)
# ===================
Player DeepInTheQ won 70 times (70.0%)
Player Randy Random won 30 times (30.0%)
We missplyed 1237 times over 1940 turns, (63.76288659793814%)

Player Randy Random won 40 times (40.0%)
Player DeepInTheQ won 60 times (60.0%)
We missplyed 1229 times over 1988 turns, (61.82092555331992%)
# ======================================
```


## V1 - MemoNerd

Knows too much and is a jerk about it.

Algorithm used : Q value learning

First try for an AI, set a value for each each possible state and choose the best one.

### Results

```md
# VS Firsty First
Player DeathCursor won 4 times (4.0%)
Player DeathCursor won 6 times (6.0%)

# VS Randy Random
Player DeathCursor won 48 times (48.0%)
Player DeathCursor won 46 times (46.0%)
```

The nerd is not that good, there is too much moves to remember.
I stoped training after 7mi saved board and during testing, 90% of the time, the AI was playing randomly because it didn't know what to do. 
