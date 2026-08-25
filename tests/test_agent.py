"""Tests for features, policy, reward shaping, sleep, and personality loading.

Two things here are worth more than the rest:

  - `test_fair_bets_are_not_scored_as_losers` pins down a real bug that was
    found by measurement: clipping the training error instead of the target
    biases the model pessimistic, because a long shot's +19 win clips while its
    -1 loss does not.
  - The reward tests pin down the PRD's named failure mode -- an agent that
    discovers permanent inaction is the safest path to a non-negative score.
"""

import datetime as dt
import random

import pytest

from sim import money
from sim.fills import YES, NO
from kalshi.universe import Market
from agent import features, sleep as sl
from agent.features import FEATURE_NAMES, FEATURE_INDEX, N_FEATURES
from agent.policy import LinearQ, Decision
from agent.reward import RewardWeights, daily_reward, trade_target
from agent.sleep import SleepSchedule
from agent.personality import load_all, Sizing


def _market(bid=4000, ask=4200, close_in=3600, volume=100.0, oi=50.0):
    import time
    m = Market("KXTEST-1")
    m.category, m.series, m.subtitle = "Sports", "KXTEST", "Someone"
    m.yes_bid, m.yes_ask = bid, ask
    m.yes_bid_size = m.yes_ask_size = 500.0
    m.volume, m.open_interest = volume, oi
    m.close_ts = (time.time() + close_in) if close_in else None
    return m


class _Candidate:
    def __init__(self, vector):
        self.features = vector


def _vector(**overrides):
    v = [0.0] * N_FEATURES
    v[FEATURE_INDEX["bias"]] = 1.0
    for name, value in overrides.items():
        v[FEATURE_INDEX[name]] = value
    return v


# ---------- features ----------

def test_entry_price_differs_by_side():
    """Buying YES at 42c and NO at 60c are different bets on the same market."""
    m = _market(bid=4000, ask=4200)
    assert features.entry_price(m, YES) == 4200
    assert features.entry_price(m, NO) == money.no_price(4000)   # 6000


def test_an_unbuyable_side_produces_no_candidate():
    m = _market(bid=0, ask=4200)          # nobody bidding, so NO cannot be bought
    assert features.entry_price(m, NO) is None
    assert features.build(m, NO) is None


def test_the_vector_is_always_the_same_length_and_order():
    """Weights are indexed positionally; a shifting vector silently corrupts them."""
    for m in (_market(), _market(volume=None, oi=None, close_in=None)):
        vector = features.build(m, YES)
        assert len(vector) == N_FEATURES
    assert features.named(features.build(_market(), YES))["bias"] == 1.0


def test_price_indicators_fire_in_the_right_bands():
    assert features.named(features.build(_market(bid=1000, ask=1500), YES))["price_longshot"] == 1.0
    assert features.named(features.build(_market(bid=4800, ask=5000), YES))["price_even"] == 1.0
    assert features.named(features.build(_market(bid=8800, ask=9000), YES))["price_favourite"] == 1.0


def test_missing_market_data_becomes_zero_not_a_hole():
    m = _market(volume=None, oi=None, close_in=None)
    named = features.named(features.build(m, YES))
    assert named["log_volume"] == 0.0
    assert named["log_time_to_close"] == 0.0


def test_with_no_memory_all_memory_features_are_zero():
    """An agent that has never seen a market genuinely knows nothing about it."""
    named = features.named(features.build(_market(), YES, memory=None))
    for name in FEATURE_NAMES:
        if name.startswith("mem_"):
            assert named[name] == 0.0


class _Belief:
    """Minimal stand-in for agent.memory.Belief."""

    def __init__(self, win_rate=0.5, roi=0.0, ewma=0.0, n=50):
        self.win_rate, self.roi, self.ewma_return, self._n = win_rate, roi, ewma, n

    def confidence(self, half=10.0):
        return self._n / (self._n + half)


def test_memory_is_reported_per_tier_not_blended():
    """Broad, series, and entity memory stay separate.

    Blending them was the bug that made a category with a -34% YES side and a
    +62% NO side look break-even, and left the agent taking the losing side for
    400 simulated days.
    """
    summary = features.summarize_memory([
        ("category_side", _Belief(win_rate=0.20, roi=-0.5)),
        ("series_side", _Belief(win_rate=0.90, roi=+1.2)),
    ])
    assert summary["broad_winrate"] < 0        # the category looks bad
    assert summary["series_winrate"] > 0       # this market type looks good
    assert summary["broad_roi"] != summary["series_roi"]


def test_the_side_aware_belief_wins_over_the_side_blind_one():
    """"Which way to bet on this" is exactly what a side-blind key cannot say."""
    summary = features.summarize_memory([
        ("series", _Belief(win_rate=0.50, roi=0.0)),
        ("series_side", _Belief(win_rate=0.95, roi=+1.5)),
    ])
    assert summary["series_winrate"] > 0.4     # the side-aware view won


def test_memory_ratios_are_clipped():
    """One 19x win must not dominate every weight it touches."""
    summary = features.summarize_memory([("series_side", _Belief(roi=50.0))])
    assert summary["series_roi"] == features.ROI_CLIP


# ---------- policy ----------

def test_a_fresh_agent_is_optimistic_enough_to_act():
    """The cold start: zero weights mean Q=0 everywhere and the agent never
    trades, never observes an outcome, and never learns."""
    q = LinearQ(optimism=0.15, epsilon=0.0)
    decision = q.select([_Candidate(_vector())])
    assert decision.acted is True
    assert decision.q > 0


def test_an_agent_with_nothing_worth_doing_skips():
    """PRD 2: choosing not to trade must remain a legitimate strategy."""
    q = LinearQ(optimism=0.0, epsilon=0.0)
    q.weights[FEATURE_INDEX["bias"]] = -0.5
    decision = q.select([_Candidate(_vector())])
    assert decision.acted is False
    assert decision.skipped_reason == "nothing_worth_doing"


def test_no_candidates_is_a_distinct_kind_of_skip():
    decision = LinearQ().select([])
    assert decision.acted is False
    assert decision.skipped_reason == "no_candidates"


def test_exploitation_takes_the_highest_scoring_candidate():
    q = LinearQ(epsilon=0.0, optimism=0.0)
    q.weights[FEATURE_INDEX["price"]] = 1.0
    best = _Candidate(_vector(price=0.9))
    decision = q.select([_Candidate(_vector(price=0.1)), best])
    assert decision.candidate is best
    assert decision.explored is False


def test_exploration_can_pick_something_other_than_the_best():
    q = LinearQ(epsilon=1.0, optimism=0.0, seed=1)
    q.weights[FEATURE_INDEX["price"]] = 1.0
    candidates = [_Candidate(_vector(price=p)) for p in (0.1, 0.5, 0.9)]
    picked = {id(q.select(candidates).candidate) for _ in range(30)}
    assert len(picked) > 1
    assert q.exploration_rate == 1.0


def test_epsilon_drives_the_exploration_rate():
    """Kenny's 0.55 and Kyle's 0.05 are this number and nothing else."""
    rates = []
    for epsilon in (0.05, 0.55):
        q = LinearQ(epsilon=epsilon, optimism=0.5, seed=4)
        for _ in range(2000):
            q.select([_Candidate(_vector()), _Candidate(_vector(price=0.5))])
        rates.append(q.exploration_rate)
    assert rates[0] < 0.12
    assert rates[1] > 0.45


def test_learning_moves_the_prediction_toward_the_outcome():
    q = LinearQ(learning_rate=0.1, optimism=0.0)
    vector = _vector(price=0.5)
    before = q.q(vector)
    for _ in range(50):
        q.update(vector, 1.0)
    assert q.q(vector) > before


def test_fair_bets_are_not_scored_as_losers():
    """The bias that error-clipping introduced, pinned down.

    In a world where every bet has exactly zero expected value, a correct
    learner converges to roughly zero. Clipping the error instead of the target
    made this converge near -0.3, which would have made every agent behave like
    Kyle and left Cartman's long-shot preference with nothing to find.
    """
    rng = random.Random(11)
    q = LinearQ(learning_rate=0.005, epsilon=1.0, optimism=0.0, seed=1)
    for _ in range(40000):
        price = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        vector = _vector(price=price)
        won = rng.random() < price                 # fair odds, zero EV
        q.update(vector, (1.0 / price - 1.0) if won else -1.0)

    for price in (0.3, 0.5, 0.9):
        assert abs(q.q(_vector(price=price))) < 0.2


def test_an_extreme_target_is_bounded_but_normal_wins_are_not():
    q = LinearQ(learning_rate=0.01, optimism=0.0)
    vector = _vector()
    q.update(vector, 999.0)                        # a sub-1c lottery ticket
    bounded = q.q(vector)
    assert 0 < bounded <= 0.01 * 20.0 + 1e-9       # clipped to TARGET_CLIP


def test_explain_decomposes_a_score_exactly():
    """The payoff for a linear model: the parts sum to the whole, no remainder."""
    q = LinearQ(optimism=0.2)
    q.weights[FEATURE_INDEX["price"]] = 0.5
    q.weights[FEATURE_INDEX["mem_series_winrate"]] = -0.3
    vector = _vector(price=0.8, mem_series_winrate=0.25)

    parts = q.explain(vector)
    assert sum(contribution for _, contribution, _, _ in parts) == pytest.approx(q.q(vector))
    assert parts[0][0] == "price"                  # largest magnitude first


def test_weights_are_readable_by_name():
    """PRD 3.4 and 14: the model must be explainable, not just functional."""
    named = LinearQ().named_weights()
    assert set(named) == set(FEATURE_NAMES)
    assert isinstance(named["bias"], float)


# ---------- persistence ----------

def test_weights_survive_a_restart(tmp_path):
    path = str(tmp_path / "w.npz")
    first = LinearQ(learning_rate=0.1, epsilon=0.3)
    for _ in range(20):
        first.update(_vector(price=0.5), 0.8)
    first.save(path)

    second = LinearQ()
    assert second.load(path) is True
    assert second.q(_vector(price=0.5)) == pytest.approx(first.q(_vector(price=0.5)))
    assert second.updates == first.updates
    assert second.epsilon == 0.3


def test_loading_a_missing_file_is_not_an_error():
    assert LinearQ().load("/nonexistent/weights.npz") is False


def test_mismatched_feature_names_are_refused(tmp_path):
    """A changed feature set must not silently misalign every weight.

    That would not raise -- it would just make a trained agent behave randomly,
    which is far worse than starting fresh.
    """
    import numpy as np
    path = str(tmp_path / "old.npz")
    np.savez(path, weights=np.zeros(3), names=np.array(["a", "b", "c"]),
             meta=np.array(['{}']))
    assert LinearQ().load(path) is False


# ---------- reward ----------

def _weights(**overrides):
    return RewardWeights(**overrides)


def test_a_profitable_day_scores_positive():
    reward, parts = daily_reward(
        _weights(), realized_pnl=money.dollars(10), trades=3, drawdown=0.0,
        largest_exposure=money.dollars(5), bankroll=money.dollars(95))
    assert reward > 0
    assert parts["pnl"] > 0


def test_losses_count_more_than_equivalent_gains():
    """Mild loss aversion (PRD 5) -- enough to teach capital preservation."""
    gain, _ = daily_reward(_weights(), realized_pnl=money.dollars(10), trades=1,
                           drawdown=0.0, largest_exposure=0, bankroll=money.dollars(100))
    loss, _ = daily_reward(_weights(), realized_pnl=money.dollars(-10), trades=1,
                           drawdown=0.0, largest_exposure=0, bankroll=money.dollars(100))
    assert abs(loss) > abs(gain)


def test_loss_aversion_stays_mild():
    """Too harsh and every bet looks bad, which is the paralysis trap."""
    assert 1.0 < RewardWeights().loss_aversion < 1.5


def test_doing_nothing_is_slightly_negative_not_free():
    """The PRD's named failure mode: an agent that discovers inaction is the
    safest route to a non-negative reward and stops trading forever."""
    reward, parts = daily_reward(
        _weights(), realized_pnl=0, trades=0, drawdown=0.0,
        largest_exposure=0, bankroll=money.dollars(100))
    assert reward < 0
    assert parts["inaction"] < 0
    assert reward > -0.05        # discouraged, not punished into forced trading


def test_holding_positions_is_not_idleness():
    """Waiting on open positions is a real strategy, not doing nothing."""
    idle, _ = daily_reward(_weights(), realized_pnl=0, trades=0, drawdown=0.0,
                           largest_exposure=0, bankroll=money.dollars(100),
                           open_positions=0)
    holding, _ = daily_reward(_weights(), realized_pnl=0, trades=0, drawdown=0.0,
                              largest_exposure=0, bankroll=money.dollars(100),
                              open_positions=4)
    assert holding > idle


def test_inaction_still_beats_a_catastrophic_day():
    """Otherwise the agent would learn that gambling wildly beats sitting out."""
    idle, _ = daily_reward(_weights(), realized_pnl=0, trades=0, drawdown=0.0,
                           largest_exposure=0, bankroll=money.dollars(100))
    disaster, _ = daily_reward(_weights(), realized_pnl=money.dollars(-40), trades=9,
                               drawdown=0.4, largest_exposure=money.dollars(40),
                               bankroll=money.dollars(60))
    assert idle > disaster


def test_an_oversized_bet_is_penalised_even_on_a_winning_day():
    """PRD 5: a reckless win must not reinforce as strongly as a well-sized one."""
    modest, _ = daily_reward(_weights(), realized_pnl=money.dollars(10), trades=1,
                             drawdown=0.0, largest_exposure=money.dollars(5),
                             bankroll=money.dollars(95))
    reckless, _ = daily_reward(_weights(), realized_pnl=money.dollars(10), trades=1,
                               drawdown=0.0, largest_exposure=money.dollars(60),
                               bankroll=money.dollars(40))
    assert modest > reckless


def test_exposure_below_the_free_fraction_is_not_penalised():
    _, parts = daily_reward(_weights(), realized_pnl=0, trades=1, drawdown=0.0,
                            largest_exposure=money.dollars(5),
                            bankroll=money.dollars(95))
    assert parts["exposure"] == 0.0


def test_kyle_is_punished_harder_than_cartman_for_the_same_reckless_day():
    """Personality expressed through the reward function, not through code."""
    agents = load_all()
    day = dict(realized_pnl=money.dollars(5), trades=4, drawdown=0.30,
               largest_exposure=money.dollars(50), bankroll=money.dollars(50))
    kyle, _ = daily_reward(agents["kyle"].reward, **day)
    cartman, _ = daily_reward(agents["cartman"].reward, **day)
    assert kyle < cartman


def test_overtrading_costs_something():
    few, _ = daily_reward(_weights(), realized_pnl=0, trades=2, drawdown=0.0,
                          largest_exposure=0, bankroll=money.dollars(100))
    many, _ = daily_reward(_weights(), realized_pnl=0, trades=60, drawdown=0.0,
                           largest_exposure=0, bankroll=money.dollars(100))
    assert many < few


def test_bankruptcy_is_a_strong_terminal_penalty():
    normal, _ = daily_reward(_weights(), realized_pnl=money.dollars(-20), trades=3,
                             drawdown=0.2, largest_exposure=money.dollars(20),
                             bankroll=money.dollars(80))
    ruined, parts = daily_reward(_weights(), realized_pnl=money.dollars(-20), trades=3,
                                 drawdown=0.2, largest_exposure=money.dollars(20),
                                 bankroll=money.dollars(80), went_bankrupt=True)
    assert ruined < normal
    assert parts["bankruptcy"] < -1


def test_a_position_held_at_ruin_trains_harder_than_an_ordinary_wipeout():
    """The daily bankruptcy penalty is a scoreboard entry -- it never reaches a
    weight. If ruin is to be learnable at all it has to land here."""
    from sim import fills
    from sim.portfolio import Portfolio

    def _total_loss(result):
        portfolio = Portfolio("stan")
        position = portfolio.open_position(
            "M", YES, fills.buy([(5000, 1000)], 10, YES), day="d1")
        position.realized_pnl = -position.cost
        position.result = result
        return position

    weights = _weights()
    ordinary = trade_target(_total_loss("loss"), weights)
    ruinous = trade_target(_total_loss("bankrupt"), weights)

    assert ruinous < ordinary
    assert ordinary - ruinous == pytest.approx(weights.ruin_penalty)


def test_agents_are_marked_differently_by_being_ruined():
    """Kyle should carry a blow-up; Cartman shrugging it off is his character."""
    agents = load_all()
    penalties = {n: p.reward.ruin_penalty for n, p in agents.items()}
    assert penalties["kyle"] == max(penalties.values())
    assert penalties["cartman"] == min(penalties.values())


def test_the_breakdown_sums_to_the_reward():
    """A single scalar is useless for diagnosing a reward loophole."""
    reward, parts = daily_reward(_weights(), realized_pnl=money.dollars(-5), trades=7,
                                 drawdown=0.15, largest_exposure=money.dollars(30),
                                 bankroll=money.dollars(70))
    assert sum(parts.values()) == pytest.approx(reward)


class _Pos:
    def __init__(self, pnl, cost, stake_fraction=0.0):
        self.realized_pnl = pnl
        self.cost = cost
        self.stake_fraction = stake_fraction


def test_the_trade_target_is_return_on_risk():
    assert trade_target(_Pos(money.dollars(5), money.dollars(10))) == pytest.approx(0.5)


def test_a_reckless_win_trains_a_smaller_target_than_a_careful_one():
    """PRD 5: a reckless win must not reinforce as strongly as a well-sized win.

    Return on risk is scale-free, so without this shaping the two are literally
    the same training example and PRD 5's risk shaping never reaches the model.
    """
    weights = RewardWeights(exposure_penalty=0.5, exposure_free_fraction=0.2)
    careful = trade_target(_Pos(money.dollars(5), money.dollars(10), 0.05), weights)
    reckless = trade_target(_Pos(money.dollars(5), money.dollars(10), 0.60), weights)
    assert careful > reckless
    assert careful == pytest.approx(0.5)     # under the free allowance


def test_losses_train_a_harsher_target_than_equivalent_wins():
    weights = RewardWeights(loss_aversion=1.5)
    win = trade_target(_Pos(money.dollars(5), money.dollars(10)), weights)
    loss = trade_target(_Pos(money.dollars(-5), money.dollars(10)), weights)
    assert abs(loss) > abs(win)


def test_kyle_and_cartman_learn_different_lessons_from_the_same_trade():
    """Their exposure penalties differ, so an identical oversized win is a
    different lesson for each. This is personality reaching the learner."""
    agents = load_all()
    position = _Pos(money.dollars(8), money.dollars(10), stake_fraction=0.5)
    kyle = trade_target(position, agents["kyle"].reward)
    cartman = trade_target(position, agents["cartman"].reward)
    assert kyle < cartman


def test_the_target_is_unshaped_when_no_weights_are_given():
    assert trade_target(_Pos(money.dollars(5), money.dollars(10), 0.9)) == pytest.approx(0.5)


def test_an_unresolved_position_has_no_training_target():
    assert trade_target(_Pos(None, money.dollars(10))) is None
    assert trade_target(_Pos(money.dollars(5), 0)) is None


# ---------- sleep ----------

def test_an_agent_is_asleep_inside_its_window():
    schedule = SleepSchedule(start_hour=1.0, duration_hours=8.0)
    assert sl.is_asleep("x", schedule, dt.datetime(2026, 8, 24, 3, 0)) is True
    assert sl.is_asleep("x", schedule, dt.datetime(2026, 8, 24, 12, 0)) is False


def test_a_window_that_wraps_past_midnight_works():
    """Cartman starts at 03:00 and Kenny might start at 23:00 -- wrapping is
    the normal case here, not an edge case."""
    schedule = SleepSchedule(start_hour=23.0, duration_hours=8.0)
    assert sl.is_asleep("x", schedule, dt.datetime(2026, 8, 24, 23, 30)) is True
    assert sl.is_asleep("x", schedule, dt.datetime(2026, 8, 24, 3, 0)) is True
    assert sl.is_asleep("x", schedule, dt.datetime(2026, 8, 24, 12, 0)) is False


def test_the_schedule_is_stable_within_a_day():
    """Otherwise "is Kenny asleep?" would flicker on every dashboard poll."""
    schedule = SleepSchedule(start_hour=4.0, duration_hours=5.0, jitter_hours=3.0)
    first = sl.schedule_for("kenny", schedule, "2026-08-24")
    for _ in range(20):
        assert sl.schedule_for("kenny", schedule, "2026-08-24") == first


def test_the_schedule_changes_between_days_when_jittered():
    schedule = SleepSchedule(start_hour=4.0, duration_hours=5.0, jitter_hours=3.0)
    starts = {sl.schedule_for("kenny", schedule, f"2026-08-{d:02d}")[0]
              for d in range(1, 15)}
    assert len(starts) > 5


def test_a_disciplined_schedule_barely_moves():
    """Kyle: the same window every day, rarely varies (PRD 8)."""
    schedule = load_all()["kyle"].sleep
    starts = [sl.schedule_for("kyle", schedule, f"2026-08-{d:02d}")[0]
              for d in range(1, 15)]
    assert max(starts) - min(starts) < 0.5


def test_kenny_sometimes_skips_sleep_entirely():
    schedule = load_all()["kenny"].sleep
    skipped = [sl.schedule_for("kenny", schedule, f"2026-{m:02d}-{d:02d}")[2]
               for m in range(1, 13) for d in range(1, 28)]
    assert any(skipped)
    assert not all(skipped)


def test_a_skipped_night_means_a_full_day_awake():
    schedule = SleepSchedule(start_hour=4.0, duration_hours=5.0, skip_probability=1.0)
    assert sl.is_asleep("x", schedule, dt.datetime(2026, 8, 24, 5, 0)) is False
    assert sl.awake_hours_on("x", schedule, "2026-08-24") == 24.0


def test_kenny_is_awake_longest_and_cartman_shortest():
    """PRD 8: awake time is a genuine confound when comparing performance."""
    agents = load_all()
    hours = {n: 24.0 - p.sleep.duration_hours for n, p in agents.items()}
    assert hours["kenny"] == max(hours.values())
    assert hours["cartman"] == min(hours.values())


# ---------- personality ----------

def test_all_four_agents_load():
    agents = load_all()
    assert set(agents) == {"stan", "kyle", "cartman", "kenny"}


def test_defaults_are_inherited_and_overrides_win():
    agents = load_all()
    assert agents["stan"].policy.epsilon == 0.15        # the default
    assert agents["kyle"].policy.epsilon == 0.05        # overridden


def test_the_personalities_are_actually_different():
    """If Cartman behaves like Stan, the config is wrong -- and there is no
    hidden code path where his greed could be lurking instead."""
    agents = load_all()
    epsilons = {n: p.policy.epsilon for n, p in agents.items()}
    assert len(set(epsilons.values())) == 4
    assert epsilons["kenny"] > epsilons["cartman"] > epsilons["stan"] > epsilons["kyle"]


def test_cartman_bets_bigger_than_kyle():
    agents = load_all()
    assert agents["cartman"].sizing.max_fraction > agents["kyle"].sizing.max_fraction * 5


def test_cartman_has_the_shortest_memory_and_kyle_the_longest():
    """"Over-repeats whatever recently worked" is one number (PRD 4)."""
    agents = load_all()
    assert agents["cartman"].memory_ewma_alpha > agents["stan"].memory_ewma_alpha
    assert agents["kyle"].memory_ewma_alpha < agents["stan"].memory_ewma_alpha


def test_kenny_has_the_widest_sizing_band():
    """Erratic sizing is the width of the distribution, not its mean."""
    agents = load_all()
    width = {n: p.sizing.max_fraction - p.sizing.min_fraction
             for n, p in agents.items()}
    assert width["kenny"] >= max(width.values())


def test_bet_sizing_respects_the_absolute_cap():
    """A backstop against a config typo emptying an agent in one trade."""
    sizing = Sizing(min_fraction=0.9, max_fraction=0.9, absolute_max_fraction=0.35)
    stake = sizing.sample(random.Random(0), money.dollars(100))
    assert stake <= money.dollars(35)


def test_bankroll_is_converted_from_dollars_to_money_units():
    assert load_all()["stan"].starting_bankroll == money.STARTING_BANKROLL


def test_every_agent_exposes_a_dashboard_summary():
    for personality in load_all().values():
        summary = personality.summary()
        assert summary["display_name"]
        assert summary["blurb"]
        assert "sleep" in summary and "reward" in summary


# ---------- tied scores must not resolve to list order ----------

def test_tied_candidates_are_not_resolved_by_list_order():
    """A fresh agent has all-zero weights, so every candidate scores exactly the
    optimism prior and the whole list is one tie. Taking the first maximum made
    its choices an artifact of how the sampler ordered the list -- seen live,
    two agents kept buying long-dated markets after sampling was reweighted
    toward fast ones, because the reweighted half was never reached."""
    q = LinearQ(epsilon=0.0, optimism=0.5, seed=1)
    candidates = [_Candidate(_vector()) for _ in range(40)]
    picked = {id(q.select(candidates).candidate) for _ in range(200)}
    assert len(picked) > 1, "always picked the same position in the list"


def test_a_genuinely_better_candidate_still_wins():
    """Tie-breaking must not become coin-flipping between unequal options."""
    q = LinearQ(epsilon=0.0, optimism=0.0, seed=2)
    q.weights[FEATURE_INDEX["price"]] = 1.0
    candidates = [_Candidate(_vector(price=0.1)) for _ in range(20)]
    best = _Candidate(_vector(price=0.9))
    candidates.insert(7, best)
    for _ in range(50):
        assert q.select(candidates).candidate is best


def test_tie_breaking_does_not_count_as_exploration():
    """Exploration is a deliberate epsilon-greedy departure from the best pick.
    Choosing among equals is still exploitation and must not inflate the
    exploration rate PRD 9 plots."""
    q = LinearQ(epsilon=0.0, optimism=0.5, seed=3)
    candidates = [_Candidate(_vector()) for _ in range(10)]
    for _ in range(30):
        assert q.select(candidates).explored is False
    assert q.exploration_rate == 0.0
