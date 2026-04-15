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
#  Indicator helpers
# ─────────────────────────────────────────────────────────────────────────────


def rsi(prices: list[float], period: int = 14) -> float | None:
    """Classic Wilder RSI. Returns 0-100 or None if insufficient data."""
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


# ─────────────────────────────────────────────────────────────────────────────
#  Trader
# ─────────────────────────────────────────────────────────────────────────────


class Trader:
    LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    EMERALD_FAIR = 10000

    # ── Tomatoes EMA config ────────────────────────────────────────────────
    # Persistent EMA-50 / EMA-200 on mid price
    K_FAST = 2.0 / (50 + 1)
    K_SLOW = 2.0 / (200 + 1)
    EMA_WARMUP = 50  # ticks before trend signal activates

    SIGNAL_SCALE = 8.0  # EMA-diff ticks → ±1 signal

    # OBI (order book imbalance) blend weight.
    # OBI = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
    # Gives fast microstructure direction on top of slow EMA trend.
    OBI_WEIGHT = 0.25  # 25% OBI + 75% EMA-trend

    # Maker quote fractions
    BASE_FRAC = 0.40  # base size each side (% of limit)
    SKEW_FRAC = 0.40  # how much trend skews between buy/sell

    # Inventory skew: shift both bid and ask by this many ticks × (pos/limit)
    # Long position → shift quotes DOWN (cheaper to sell, dearer to buy)
    # Short position → shift quotes UP
    INV_SKEW_TICKS = 2

    # Taker de-risk: shed inventory when position is wrong-way to strong signal
    TAKER_POS_THRESH = 0.50
    TAKER_SIG_THRESH = 0.75
    TAKER_REDUCE_FRAC = 0.25

    # RSI filter: suppress quoting in direction of exhaustion
    RSI_PERIOD = 14
    RSI_OB = 68  # overbought → trim long quotes
    RSI_OS = 32  # oversold   → trim short quotes
    RSI_MAX_HIST = 20  # keep only 20 prices in traderData (tiny budget)

    def run(self, state: TradingState):
        result = {}

        # ── Deserialise persistent state ──────────────────────────────────────
        try:
            memory = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            memory = {}

        tom_mid_hist: list[float] = memory.get("tom_mid_hist", [])

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = self.LIMITS.get(product, 80)

            # ═══════════════════════════════════════════════════════════════
            #  EMERALDS  – pure fair-value market making
            # ═══════════════════════════════════════════════════════════════
            if product == "EMERALDS":
                fair = self.EMERALD_FAIR

                # ── Step 1: take mispriced taker orders ───────────────────
                if order_depth.sell_orders:
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if ask_price < fair:
                            available = -order_depth.sell_orders[ask_price]
                            qty = min(available, limit - position)
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
                            qty = min(available, limit + position)
                            if qty > 0:
                                orders.append(Order(product, bid_price, -qty))
                                position -= qty
                        else:
                            break

                # ── Step 2: maker quotes always at fair±1 ─────────────────
                # FIX vs v1: previously gated on `our_bid < fair AND our_ask > fair`
                # which silently skipped quotes when spread was tight (e.g. 9999/10001
                # → our_bid=10000 which fails the strict < check). Now we always post
                # at 9999/10001, clamping penny-in so we never cross fair.
                #
                # Result: maximum fill volume while never paying above fair.
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    # Penny in but cap/floor at fair±1
                    our_bid = min(best_bid + 1, fair - 1)
                    our_ask = max(best_ask - 1, fair + 1)

                    if our_bid < our_ask:  # sanity: no locked/crossed market
                        buy_room = limit - position
                        sell_room = limit + position
                        if buy_room > 0:
                            orders.append(Order(product, our_bid, buy_room))
                        if sell_room > 0:
                            orders.append(Order(product, our_ask, -sell_room))

            # ═══════════════════════════════════════════════════════════════
            #  TOMATOES  – trend-following + OBI + inventory skew
            # ═══════════════════════════════════════════════════════════════
            elif product == "TOMATOES":
                if not order_depth.sell_orders or not order_depth.buy_orders:
                    result[product] = orders
                    continue

                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2.0

                # ── 1. Update persistent EMA state ────────────────────────
                fast_ema = memory.get("tom_fast", mid_price)
                slow_ema = memory.get("tom_slow", mid_price)
                tick_count = memory.get("tom_ticks", 0)

                fast_ema = mid_price * self.K_FAST + fast_ema * (1 - self.K_FAST)
                slow_ema = mid_price * self.K_SLOW + slow_ema * (1 - self.K_SLOW)
                tick_count += 1

                memory["tom_fast"] = fast_ema
                memory["tom_slow"] = slow_ema
                memory["tom_ticks"] = tick_count

                # ── 2. Update mid-price history for RSI ───────────────────
                tom_mid_hist.append(mid_price)
                if len(tom_mid_hist) > self.RSI_MAX_HIST + 2:
                    tom_mid_hist = tom_mid_hist[-(self.RSI_MAX_HIST + 2) :]
                memory["tom_mid_hist"] = tom_mid_hist

                # ── 3. EMA trend signal ───────────────────────────────────
                if tick_count < self.EMA_WARMUP:
                    ema_trend = 0.0
                else:
                    ema_diff = fast_ema - slow_ema
                    ema_trend = max(-1.0, min(1.0, ema_diff / self.SIGNAL_SCALE))

                # ── 4. Order Book Imbalance (OBI) ─────────────────────────
                # NEW: fast microstructure signal.
                # OBI > 0 → more buy pressure at touch → expect upward tick
                # OBI < 0 → more sell pressure → expect downward tick
                # Blended with slow EMA trend for composite direction signal.
                best_bid_vol = order_depth.buy_orders.get(best_bid, 0)
                best_ask_vol = -order_depth.sell_orders.get(best_ask, 0)
                total_touch_vol = best_bid_vol + best_ask_vol
                obi = (
                    (best_bid_vol - best_ask_vol) / total_touch_vol
                    if total_touch_vol > 0
                    else 0.0
                )

                # Composite signal: mostly EMA trend, flavoured by OBI
                signal = max(
                    -1.0,
                    min(
                        1.0,
                        (1 - self.OBI_WEIGHT) * ema_trend + self.OBI_WEIGHT * obi,
                    ),
                )

                # ── 5. RSI filter ─────────────────────────────────────────
                # NEW: when RSI is extreme, fade the signal in that direction
                # (don't add longs when overbought; don't add shorts when oversold)
                rsi_val = rsi(tom_mid_hist, self.RSI_PERIOD)
                rsi_buy_mult = 1.0
                rsi_sell_mult = 1.0
                if rsi_val is not None:
                    if rsi_val > self.RSI_OB:
                        # Overbought: shrink buy fraction
                        excess = (rsi_val - self.RSI_OB) / (100 - self.RSI_OB)
                        rsi_buy_mult = max(0.0, 1.0 - excess)
                    elif rsi_val < self.RSI_OS:
                        # Oversold: shrink sell fraction
                        excess = (self.RSI_OS - rsi_val) / self.RSI_OS
                        rsi_sell_mult = max(0.0, 1.0 - excess)

                # ── 6. Taker de-risk (last resort) ────────────────────────
                if (
                    position > limit * self.TAKER_POS_THRESH
                    and signal < -self.TAKER_SIG_THRESH
                ):
                    target = int(limit * self.TAKER_POS_THRESH)
                    shed_qty = min(
                        position - target, int(limit * self.TAKER_REDUCE_FRAC)
                    )
                    for bid_price in sorted(
                        order_depth.buy_orders.keys(), reverse=True
                    ):
                        if shed_qty <= 0:
                            break
                        avail = order_depth.buy_orders[bid_price]
                        qty = min(avail, shed_qty, limit + position)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            position -= qty
                            shed_qty -= qty

                elif (
                    position < -limit * self.TAKER_POS_THRESH
                    and signal > self.TAKER_SIG_THRESH
                ):
                    target = -int(limit * self.TAKER_POS_THRESH)
                    shed_qty = min(
                        -position + target, int(limit * self.TAKER_REDUCE_FRAC)
                    )
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if shed_qty <= 0:
                            break
                        avail = -order_depth.sell_orders[ask_price]
                        qty = min(avail, shed_qty, limit - position)
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            position += qty
                            shed_qty -= qty

                # ── 7. Trend-skewed maker quotes + inventory skew ─────────
                # NEW: inventory_skew shifts both bid and ask by up to ±INV_SKEW_TICKS
                # proportional to (position / limit). When long → quotes shift DOWN
                # (cheaper ask, dearer bid suppressed) → fills naturally flatten us.
                # When short → quotes shift UP → fills naturally cover us.
                #
                # This replaces the asymmetric "never short" behavior seen in v1.

                if spread < 2:
                    # Spread collapsed — no maker edge
                    result[product] = orders
                    continue

                inventory_ratio = position / limit  # ∈ [-1, +1]
                inv_skew = -round(self.INV_SKEW_TICKS * inventory_ratio)
                # inv_skew > 0 when short (shift up), < 0 when long (shift down)

                our_bid = best_bid + 1 + inv_skew
                our_ask = best_ask - 1 + inv_skew

                if our_bid >= our_ask:
                    result[product] = orders
                    continue

                # Signal + RSI combined quote fractions
                buy_frac = (
                    max(0.0, self.BASE_FRAC + self.SKEW_FRAC * signal) * rsi_buy_mult
                )
                sell_frac = (
                    max(0.0, self.BASE_FRAC - self.SKEW_FRAC * signal) * rsi_sell_mult
                )

                buy_room = limit - position
                sell_room = limit + position

                buy_qty = min(int(limit * buy_frac), buy_room)
                sell_qty = min(int(limit * sell_frac), sell_room)

                if buy_qty > 0:
                    orders.append(Order(product, our_bid, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, our_ask, -sell_qty))

            result[product] = orders

        # ── Serialise persistent state (compact — stays under traderData budget) ──
        traderData = json.dumps(
            {
                "tom_fast": memory.get("tom_fast", 0.0),
                "tom_slow": memory.get("tom_slow", 0.0),
                "tom_ticks": memory.get("tom_ticks", 0),
                "tom_mid_hist": tom_mid_hist[-(self.RSI_MAX_HIST + 2) :],
            },
            separators=(",", ":"),
        )

        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
