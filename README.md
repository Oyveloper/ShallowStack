# Shallowstack

This is a school project where we implement a simplified version of the DeepStack project for playing Heads Up No Limit Texas Hold´em (HUNLHE). We obviously have less resources, so some simplifications are made.

## Simplifications

### Betting

To simplify both the implementation and search space I have opted to simplify how betting works. This is done by only allowing the following bets:

| Name      | Amount      |
| --------- | ----------- |
| SMALL_BET |  10         |
| BIG_BET   | 100         |
| ALL_IN    |  everything |
