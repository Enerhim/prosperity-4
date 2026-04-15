from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import List, Any
import json
import math


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ─────────────────────────────────────────────────────────────────────────────
#  Indicator helpers (stateless pure functions — safe to call every tick)
# ─────────────────────────────────────────────────────────────────────────────


def ema(prices: list[float], period: int) -> float | None:
    """Exponential moving average over the last `period` prices.
    Returns None if not enough data yet."""
    if len(prices) < period:
        return None
    k = 2.0 / (period + 1)
    val = prices[-period]  # seed with the oldest value in window
    for p in prices[-period + 1 :]:
        val = p * k + val * (1 - k)
    return val


def rsi(prices: list[float], period: int = 14) -> float | None:
    """Classic Wilder RSI. Returns 0-100 or None if not enough data."""
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def zscore(prices: list[float], window: int) -> float | None:
    """Z-score of the latest price vs a rolling mean ± std."""
    if len(prices) < window:
        return None
    window_prices = prices[-window:]
    mean = sum(window_prices) / window
    variance = sum((p - mean) ** 2 for p in window_prices) / window
    std = math.sqrt(variance)
    if std < 1e-6:
        return 0.0
    return (prices[-1] - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
#  Trader
# ─────────────────────────────────────────────────────────────────────────────


class Trader:
    LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    EMERALD_FAIR = 10000

    # ── Tomatoes signal config ────────────────────────────────────────────────
    # EMA crossover (momentum)
    TOM_FAST_EMA = 8  # fast window (ticks)
    TOM_SLOW_EMA = 21  # slow window (ticks)

    # RSI (mean reversion / confirmation)
    TOM_RSI_PERIOD = 14
    TOM_RSI_OB = 65  # overbought → don't add longs / look to short
    TOM_RSI_OS = 35  # oversold   → don't add shorts / look to long

    # Z-score (secondary mean-reversion filter)
    TOM_ZSCORE_WIN = 20
    TOM_ZSCORE_THR = 2.0  # |z| > threshold ⇒ mean-reversion regime

    # How many ticks of history to keep (keep small to respect traderData budget)
    TOM_MAX_HISTORY = 60

    # Minimum spread before we trade at all
    TOM_MIN_SPREAD = 4

    # Position fraction per signal strength (0.33 = 1/3 of limit per "unit")
    TOM_BASE_FRAC = 0.40

    def run(self, state: TradingState):
        result = {}

        # ── Deserialise persistent state ──────────────────────────────────────
        try:
            memory = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            memory = {}

        tom_prices: list[float] = memory.get("tom_prices", [])

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = self.LIMITS.get(product, 80)

            # ── EMERALDS ──────────────────────────────────────────────────────
            if product == "EMERALDS":
                fair = self.EMERALD_FAIR

                # Take any mispriced orders (free edge)
                if order_depth.sell_orders:
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if ask_price < fair:
                            available = -order_depth.sell_orders[ask_price]
                            room = limit - position
                            qty = min(available, room)
                            if qty > 0:
                                orders.append(Order(product, ask_price, qty))
                                position += qty
                        else:
                            break

                if order_depth.buy_orders:
                    for bid_price in sorted(
                        order_depth.buy_orders.keys(), reverse=True
                    ):
                        if bid_price > fair:
                            available = order_depth.buy_orders[bid_price]
                            room = limit + position
                            qty = min(available, room)
                            if qty > 0:
                                orders.append(Order(product, bid_price, -qty))
                                position -= qty
                        else:
                            break

                # Quote just inside best bid/ask for queue priority
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    our_bid = best_bid + 1
                    our_ask = best_ask - 1

                    if our_bid < fair and our_ask > fair:
                        buy_room = limit - position
                        sell_room = limit + position
                        if buy_room > 0:
                            orders.append(Order(product, our_bid, buy_room))
                        if sell_room > 0:
                            orders.append(Order(product, our_ask, -sell_room))

            # ── TOMATOES  (trend-following + mean-reversion) ──────────────────
            elif product == "TOMATOES":
                if not order_depth.sell_orders or not order_depth.buy_orders:
                    result[product] = orders
                    continue

                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2.0

                # ── 1. Update running EMA state ───────────────────────────────
                # Key: we persist fast_ema and slow_ema VALUES, not raw prices.
                # This means EMA(200) actually behaves as a 200-tick EMA across
                # the entire trading day, not a fresh calculation each tick.
                K_FAST = 2.0 / (50 + 1)  # EMA-50:  responsive to recent drift
                K_SLOW = 2.0 / (200 + 1)  # EMA-200: slow structural trend

                fast_ema = memory.get("tom_fast", mid_price)
                slow_ema = memory.get("tom_slow", mid_price)
                tick_count = memory.get("tom_ticks", 0)

                fast_ema = mid_price * K_FAST + fast_ema * (1 - K_FAST)
                slow_ema = mid_price * K_SLOW + slow_ema * (1 - K_SLOW)
                tick_count += 1

                memory["tom_fast"] = fast_ema
                memory["tom_slow"] = slow_ema
                memory["tom_ticks"] = tick_count

                # ── 2. Compute trend signal ───────────────────────────────────
                # During warmup the EMAs are both seeded with mid_price, so the
                # diff is ~0. Zero it explicitly for the first 50 ticks anyway.
                #
                # SIGNAL_SCALE = 8: a 8-tick EMA diff = full ±1 signal.
                # For a 0.015 tick/tick drift (75 ticks over 5000 ticks),
                # steady-state EMA diff ≈ 75 × 0.015 ≈ 1.1 ticks → signal ~0.14.
                # Sharper moves (0.1 tick/tick) → diff ~7.5 → signal ~0.94.
                # This keeps us proportional and avoids binary on/off behaviour.

                SIGNAL_SCALE = 8.0

                if tick_count < 50:
                    trend = 0.0
                else:
                    ema_diff = fast_ema - slow_ema
                    trend = max(-1.0, min(1.0, ema_diff / SIGNAL_SCALE))

                logger.print(
                    f"TOM mid={mid_price:.1f} "
                    f"fast={fast_ema:.2f} slow={slow_ema:.2f} "
                    f"diff={fast_ema - slow_ema:.2f} "
                    f"trend={trend:.3f} pos={position}"
                )

                # ── 3. Taker de-risk (last resort only) ──────────────────────
                # Conditions: position is >50% of limit ON THE WRONG SIDE of a
                # strong trend (|trend| > 0.75). We shed up to 25% of limit via
                # taker orders to stop the bleeding, then let maker quotes do the
                # rest. This fires rarely — only when we've built up inventory
                # before the trend reversed sharply.

                TAKER_POS_THRESH = 0.50  # must be >50% wrong-way to trigger
                TAKER_SIG_THRESH = 0.75  # signal must be strong
                TAKER_REDUCE_FRAC = 0.25  # shed up to 25% of limit

                if position > limit * TAKER_POS_THRESH and trend < -TAKER_SIG_THRESH:
                    # Long but strongly bearish: hit the bid
                    target = int(limit * TAKER_POS_THRESH)  # de-risk down to threshold
                    shed_qty = min(position - target, int(limit * TAKER_REDUCE_FRAC))
                    shed_qty = max(0, shed_qty)
                    for bid_price in sorted(
                        order_depth.buy_orders.keys(), reverse=True
                    ):
                        if shed_qty <= 0:
                            break
                        avail = order_depth.buy_orders[bid_price]
                        room = limit + position
                        qty = min(avail, shed_qty, room)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            position -= qty
                            shed_qty -= qty

                elif position < -limit * TAKER_POS_THRESH and trend > TAKER_SIG_THRESH:
                    # Short but strongly bullish: lift the ask
                    target = -int(limit * TAKER_POS_THRESH)
                    shed_qty = min(-position + target, int(limit * TAKER_REDUCE_FRAC))
                    shed_qty = max(0, shed_qty)
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if shed_qty <= 0:
                            break
                        avail = -order_depth.sell_orders[ask_price]
                        room = limit - position
                        qty = min(avail, shed_qty, room)
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            position += qty
                            shed_qty -= qty

                # ── 4. Trend-skewed maker quotes ──────────────────────────────
                # We ALWAYS quote as maker (bid+1 / ask-1 for queue priority).
                # The trend signal shifts the volume between buy side and sell side.
                #
                # trend = +1 (strong uptrend):  buy_frac=0.8, sell_frac=0.0
                # trend =  0 (neutral):          buy_frac=0.4, sell_frac=0.4
                # trend = -1 (strong downtrend): buy_frac=0.0, sell_frac=0.8
                #
                # Over time, asymmetric fill probability builds up a position
                # in the trend direction — we earn the 13-tick spread on every
                # fill rather than paying it.

                if spread < 3:
                    # Spread collapsed — no maker edge, skip
                    result[product] = orders
                    continue

                our_bid = best_bid + 1
                our_ask = best_ask - 1

                if our_bid >= our_ask:
                    result[product] = orders
                    continue

                BASE_FRAC = 0.40
                SKEW_FRAC = 0.40

                buy_frac = max(0.0, BASE_FRAC + SKEW_FRAC * trend)
                sell_frac = max(0.0, BASE_FRAC - SKEW_FRAC * trend)

                buy_room = limit - position
                sell_room = limit + position

                buy_qty = min(int(limit * buy_frac), buy_room)
                sell_qty = min(int(limit * sell_frac), sell_room)

                if buy_qty > 0:
                    orders.append(Order(product, our_bid, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, our_ask, -sell_qty))

            result[product] = orders

        # ── Serialise only what we need (3 floats — tiny traderData) ─────────────
        traderData = json.dumps(
            {
                "tom_fast": memory.get("tom_fast", 0.0),
                "tom_slow": memory.get("tom_slow", 0.0),
                "tom_ticks": memory.get("tom_ticks", 0),
            },
            separators=(",", ":"),
        )

        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
