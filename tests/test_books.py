"""Tests for the depth-tier book registry and connection-level gap detection.

The case that matters most is `test_interleaved_markets_are_not_phantom_gaps`.
Kalshi's orderbook `seq` is one counter for the whole subscription, so a
per-market gap check -- which is what the ported `OrderBook` does, correctly, for
its original single-market use -- reports a desync on nearly every delta once
more than one market is subscribed. A live 10-minute run logged 111,574 such
phantom gaps before this module existed.
"""

from kalshi.books import BookRegistry


def snapshot(seq, ticker="A", yes="0.4700", no="0.5200"):
    return {"type": "orderbook_snapshot", "seq": seq, "msg": {
        "market_ticker": ticker,
        "yes_dollars_fp": [[yes, "300.00"]],
        "no_dollars_fp": [[no, "200.00"]]}}


def delta(seq, ticker="A", price="0.4700", size="10.00", side="yes"):
    return {"type": "orderbook_delta", "seq": seq, "msg": {
        "market_ticker": ticker, "price_dollars": price,
        "delta_fp": size, "side": side}}


def _synced(tickers=("A",), start=1):
    """A registry with a fresh snapshot for each ticker, sequence intact."""
    r = BookRegistry()
    for i, ticker in enumerate(tickers):
        r.on_message(snapshot(start + i, ticker))
    return r


# ---------- the phantom-gap regression ----------

def test_interleaved_markets_are_not_phantom_gaps():
    """One counter, many markets: A/B/A/B interleaving is perfectly in sequence.

    Under a per-market check, market A would see 1 -> 3 -> 5 and report a gap
    every single time, even though nothing was missed.
    """
    r = _synced(("A", "B"))          # snapshots take seq 1 and 2
    assert r.on_message(delta(3, "A")) is False
    assert r.on_message(delta(4, "B")) is False
    assert r.on_message(delta(5, "A")) is False
    assert r.on_message(delta(6, "B")) is False
    assert r.gaps == 0


def test_a_real_gap_is_detected():
    r = _synced(("A",))
    assert r.on_message(delta(2, "A")) is False
    assert r.on_message(delta(4, "A")) is True     # seq 3 was missed
    assert r.gaps == 1


def test_a_gap_invalidates_every_book_not_just_one():
    """The sequence number does not say which market lost a message.

    So a gap makes the whole depth tier suspect. Affordable only because the
    depth tier is small -- another reason not to put all 46k active markets in it.
    """
    r = _synced(("A", "B", "C"))
    r.on_message(delta(10, "A"))                   # jump: gap
    assert all(not book.synced for book in r.books.values())
    assert r.get("A") is None and r.get("B") is None


def test_a_gap_notifies_the_resync_callback_with_every_ticker():
    resynced = []
    r = BookRegistry(on_desync=resynced.append)
    for i, t in enumerate(("A", "B")):
        r.on_message(snapshot(1 + i, t))
    r.on_message(delta(99, "A"))
    assert len(resynced) == 1
    assert set(resynced[0]) == {"A", "B"}


# ---------- snapshots ----------

def test_new_subscriptions_mid_stream_do_not_desync_everything():
    """Snapshots share the connection-wide sequence; they do not restart it.

    Agents pull new markets into depth continuously. Treating each new batch of
    snapshots as a sequence reset made the next delta for an existing market
    look like a gap, and on a live run that left 3,113 books held and ZERO
    synced -- the depth tier permanently stale, with every fill silently falling
    back to top-of-book.
    """
    r = _synced(("A", "B"))              # snapshots at seq 1, 2
    assert r.on_message(delta(3, "A")) is False

    # A third market is subscribed; its snapshot continues the same sequence.
    assert r.on_message(snapshot(4, "C")) is False
    assert r.gaps == 0

    # And the existing books are still usable.
    assert r.on_message(delta(5, "B")) is False
    assert r.get("A") is not None and r.get("B") is not None
    assert r.get("C") is not None


def test_a_snapshot_arriving_on_a_gap_still_rebases_its_own_market():
    """The snapshot is ground truth for its market whatever happened around it."""
    r = _synced(("A", "B"))
    assert r.on_message(snapshot(99, "A")) is True    # gap: 2 -> 99
    assert r.get("A") is not None                     # A was re-based
    assert r.get("B") is None                         # B is now suspect


def test_a_snapshot_makes_a_book_readable():
    r = _synced(("A",))
    book = r.get("A")
    assert book is not None
    assert book.best_yes_bid() == 0.47
    assert r.has_depth("A") is True


def test_re_snapshotting_after_a_gap_restores_the_book():
    r = _synced(("A",))
    r.on_message(delta(50, "A"))          # gap
    assert r.get("A") is None
    r.on_message(snapshot(1, "A"))        # fresh snapshot
    assert r.get("A") is not None


# ---------- deltas ----------

def test_a_delta_mutates_the_ladder():
    r = _synced(("A",))
    before = r.get("A").yes_bid_total()
    r.on_message(delta(2, "A", price="0.4700", size="10.00"))
    assert r.get("A").yes_bid_total() == before + 10.0


def test_a_delta_for_an_unknown_market_is_ignored_not_an_error():
    """Deltas arrive in the window between subscribing and the snapshot."""
    r = _synced(("A",))
    assert r.on_message(delta(2, "UNKNOWN")) is False
    assert "UNKNOWN" not in r.books


def test_a_delta_for_a_desynced_book_is_dropped():
    """Applying deltas to a book we know is stale would corrupt it silently."""
    r = _synced(("A",))
    r.on_message(delta(50, "A"))                  # gap desyncs A
    r.on_message(delta(51, "A", size="99.00"))    # in sequence now, but stale
    assert r.get("A") is None                     # still not readable


def test_deltas_advance_the_counter_for_their_subscription():
    r = _synced(("A", "B"))
    r.on_message(delta(3, "A"))
    r.on_message(delta(4, "B"))
    assert r.last_seq[None] == 4


def test_each_subscription_has_its_own_counter():
    """Kalshi gives every subscribe command its own sid and its own sequence.

    A second subscription starting at seq 1 must not look like the first one
    jumping backwards -- that misreading left a live run with 3,000+ books held
    and zero synced.
    """
    r = BookRegistry()
    r.on_message(dict(snapshot(1, "A"), sid=1))
    r.on_message(dict(delta(2, "A"), sid=1))
    # A different subscription, its own counter starting over.
    assert r.on_message(dict(snapshot(1, "B"), sid=7)) is False
    assert r.gaps == 0
    assert r.get("A") is not None and r.get("B") is not None


def test_a_subscribed_acknowledgement_consumes_a_sequence_number():
    """It does, and swallowing it makes the next data frame look like a loss."""
    r = BookRegistry()
    r.on_message(dict(snapshot(1, "A"), sid=1))
    r.on_message({"type": "subscribed", "sid": 1, "seq": 2})
    assert r.on_message(dict(delta(3, "A"), sid=1)) is False
    assert r.gaps == 0


def test_another_channels_sequence_does_not_disturb_the_books():
    """The broad tier runs its own sids; those counters are unrelated."""
    r = BookRegistry()
    r.on_message(dict(snapshot(100, "A"), sid=1))
    r.on_message({"type": "ticker", "sid": 2, "seq": 1})
    assert r.on_message(dict(delta(101, "A"), sid=1)) is False
    assert r.gaps == 0
    assert r.get("A") is not None


# ---------- lifecycle ----------

def test_reset_drops_everything_for_a_reconnect():
    """A reconnect invalidates every book: the new connection restarts at 1."""
    r = _synced(("A", "B"))
    r.reset()
    assert r.books == {}
    assert r.last_seq == {}


def test_forget_drops_only_the_named_markets():
    r = _synced(("A", "B"))
    r.forget(["A"])
    assert "A" not in r.books and "B" in r.books


def test_first_message_on_a_connection_is_never_a_gap():
    r = BookRegistry()
    assert r.on_message(delta(500, "A")) is False
    assert r.gaps == 0


# ---------- reporting ----------

def test_stats_distinguishes_held_books_from_synced_ones():
    r = _synced(("A", "B"))
    r.on_message(delta(50, "A"))            # gap desyncs both
    s = r.stats()
    assert s["books"] == 2
    assert s["synced"] == 0
    assert s["gaps"] == 1


def test_age_tracks_when_a_book_last_changed():
    r = _synced(("A",))
    assert r.age_of("A") is not None
    assert r.age_of("NEVER-SEEN") is None
