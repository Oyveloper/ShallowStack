# Shallowstack

This is a school project where we implement a simplified version of the DeepStack project for playing Heads Up No Limit Texas Hold´em (HUNLHE). We obviously have less resources, so some simplifications are made.

## Running the system

Everything can be run through the `main.py` script. Use `python main.py --help` for more details
There are also some nice shorthand commands defined in the makefile.
For running a game where you can play against the resolve agent, simply use `make`.

## Configuration

Most of the configuration is done in the `config.ini` file, and each section is accessed from the `config.py` module.

## Simplifications

### Betting

To simplify both the implementation and search space I have opted to simplify how betting works. This is done by only allowing the following bets:

| Name        | Amount      |
| ----------- | ----------- |
| SMALL_BLIND |  5          |
| BIG_BLIND   | 10          |
| ALL_IN      |  everything |

The system also only allows for a set amount of raises in each stage.
This number has been set to 2, but can be changed in the config.

### Avg Pot size

The average pot size is used in evaluations to be ale to calculate the relative size of a given pot, which gives higher value to higher winnings.
Here the average pot size used is 200 (for 2-player games).
This value is a bit arbitrary, after experience with playing simulated games.
