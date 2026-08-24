from kalshi import orderbook
from kalshi.orderbook import OrderBook, price_key


def snap(seq=1):
    return {"type": "orderbook_snapshot", "seq": seq, "msg": {
        "market_ticker": "M",
        "yes_dollars_fp": [["0.4700", "300.00"], ["0.4600", "100.00"]],
        "no_dollars_fp": [["0.5200", "200.00"], ["0.5100", "60.00"]],
    }}


def delta(seq, price, d, side):
    return {"type": "orderbook_delta", "seq": seq, "msg": {
        "market_ticker": "M", "price_dollars": price, "delta_fp": d, "side": side}}


def test_price_key():
    assert price_key("0.5100") == 5100
    assert price_key(0.001) == 10


def test_snapshot_sets_levels_and_seq():
    b = OrderBook()
    b.apply_snapshot(snap(seq=1))
    assert b.synced is True
    assert b.last_seq == 1
    assert b.best_yes_bid() == 0.47
    # best yes ask = 1 - highest no bid (0.52) = 0.48
    assert round(b.best_yes_ask(), 4) == 0.48
    assert b.yes_bid_total() == 400.0
    assert b.yes_ask_total() == 260.0


def test_delta_applies_and_advances_seq():
    b = OrderBook()
    b.apply_snapshot(snap(seq=1))
    ok = b.apply_delta(delta(2, "0.4700", "50.00", "yes"))
    assert ok is True
    assert b.last_seq == 2
    assert b.yes[price_key("0.4700")] == 350.0


def test_delta_to_zero_removes_level():
    b = OrderBook()
    b.apply_snapshot(snap(seq=1))
    b.apply_delta(delta(2, "0.4600", "-100.00", "yes"))
    assert price_key("0.4600") not in b.yes


def test_seq_gap_returns_false():
    b = OrderBook()
    b.apply_snapshot(snap(seq=1))
    ok = b.apply_delta(delta(4, "0.4700", "10.00", "yes"))  # skipped 2,3
    assert ok is False  # caller must resync


def test_float_residue_does_not_leave_phantom_level():
    # Deltas that net to exactly zero must remove the level. Naive float addition
    # (0.10 + 0.20 - 0.30) leaves a ~5e-17 residue that would strand a phantom
    # level and can cross the book; rounding to 2 decimals must prevent that.
    b = OrderBook()
    b.apply_snapshot({"seq": 1, "msg": {"yes_dollars_fp": [], "no_dollars_fp": []}})
    for i, d in enumerate(["0.10", "0.20", "-0.30"], start=2):
        b.apply_delta({"seq": i, "msg": {
            "market_ticker": "M", "price_dollars": "0.3000", "delta_fp": d, "side": "yes"}})
    assert price_key("0.3000") not in b.yes


def test_levels_sorted():
    b = OrderBook()
    b.apply_snapshot(snap(seq=1))
    bids = b.yes_bid_levels(5)
    assert bids[0][0] == 0.47 and bids[1][0] == 0.46  # descending
    asks = b.yes_ask_levels(5)
    assert asks[0][0] < asks[1][0]  # ascending
