# Chronological 15-day public-data backtest

This tests risk controls on a checksum-validated executed-trade log; it does not claim to reproduce the user's DBot entry universe.

- Raw exits: 190
- Reconstructed episodes: 148
- Complete HKT days: 13
- Out-of-sample days: 10
- Initial equity: $50
- Requested target: $2 net per day

## Base scenario: 2% round-trip execution cost

- Net: $-1.4304
- Average/day: $-0.1430
- Maximum drawdown: 2.86%
- Trades: 69
- Win rate: 27.54%
- Net excluding top three winners: $-1.8139
- Target met: **False**

The per-day parameter choice uses only earlier complete days. Hindsight results are labeled separately and are feasibility ceilings, not validation.
