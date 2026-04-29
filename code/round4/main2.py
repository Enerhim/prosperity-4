import json
import math
import statistics
from typing import Any, Dict, List, Optional
 
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)
import jsonpickle
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Black-Scholes helpers  (r = 0, European call)
# ══════════════════════════════════════════════════════════════════════════════
 
def _ncdf(x: float) -> float:
    """Standard normal CDF via erf approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
 
 
def _bs_call(S: float, K: float, T_days: float, sigma_annual: float) -> float:
    """Black-Scholes call price.  r = 0,  T in game-days,  σ annualised."""
    T = T_days / 252.0
    if T <= 1e-9 or sigma_annual < 1e-9 or S <= 0.0 or K <= 0.0:
        return max(0.0, S - K)
    sv = sigma_annual * math.sqrt(T)
    d1 = math.log(S / K) / sv + 0.5 * sv
    d2 = d1 - sv
    return S * _ncdf(d1) - K * _ncdf(d2)
 
 
def _bs_delta(S: float, K: float, T_days: float, sigma_annual: float) -> float:
    """Delta of a European call under B-S (r = 0)."""
    T = T_days / 252.0
    if T <= 1e-9 or sigma_annual < 1e-9 or S <= 0.0 or K <= 0.0:
        return 1.0 if S > K else 0.0
    sv = sigma_annual * math.sqrt(T)
    d1 = math.log(S / K) / sv + 0.5 * sv
    return _ncdf(d1)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Logger  (do not modify)
# ══════════════════════════════════════════════════════════════════════════════
 
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
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
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
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Trader
# ══════════════════════════════════════════════════════════════════════════════
 
class Trader:
 
    # ── Exchange-enforced position limits ───────────────────────────────────
    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":        200,
        "VELVETFRUIT_EXTRACT":  200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
        "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }
 
    STRIKES: Dict[str, int] = {
        "VEV_4000": 4000, "VEV_4500": 4500,
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
        "VEV_6000": 6000, "VEV_6500": 6500,
    }
 
    # ── Regime priors from research ─────────────────────────────────────────
    # VEV oscillates 5175–5300  →  mid = 5237.5
    # HP  oscillates 9880–10080 →  mid = 9980.0
    # Spread equilibrium: 5237.5 − 0.5 × 9980 = 247.5
    _VEV_REGIME_MID: float = 5_237.5
    _HP_REGIME_MID:  float = 9_980.0
    _VEV_LO: float = 5_175.0
    _VEV_HI: float = 5_300.0
    _HP_LO:  float = 9_880.0
    _HP_HI:  float = 10_080.0
    _SPREAD_EQ: float = 247.5
 
    # ── Internal risk caps (well below exchange limits) ─────────────────────
    # VEV: 65% of exchange limit (200) = 130
    # HP:  hard cap at 20 because the 30-lot/tick ceiling means large positions
    #      cannot be exited quickly without severe market impact.
    # Options (near-ATM/OTM): 50 per strike, kept small to limit gamma risk
    # Options (deep-ITM): 40 per strike, used only vs intrinsic value mispricing
    _VEV_MAX_POS:  int = 130
    _HP_MAX_POS:   int = 12
    _OPT_ITM_MAX:  int = 40   # strikes ≤ 5000
    _OPT_NEAR_MAX: int = 50   # strikes 5100–5500
    # Deep OTM 6000/6500 → skipped entirely (per research: dead strikes)
 
    # ── Market-making parameters ────────────────────────────────────────────
    # Aggressive entry: take when price > this many ticks from fair
    _SPOT_AGGR_TICKS: int = 3
    # Passive quote offset from fair (1 tick inside)
    _SPOT_PASSIVE_OFFSET: int = 1
    # Passive order size per side (kept small to preserve fill-rate)
    _VEV_MM_SIZE: int = 15
    _HP_MM_SIZE:  int = 4   # smaller due to illiquidity
 
    # ── Pairs-trading parameters ────────────────────────────────────────────
    _PAIRS_RATIO:   float = 0.5   # VEV price ≈ 0.5 × HP price
    _PAIRS_WINDOW:  int   = 50    # rolling observations for z-score
    _PAIRS_ENTER_Z: float = 2.5   # z-score threshold to open
    _PAIRS_EXIT_Z:  float = 0.8   # z-score threshold to close
    _PAIRS_SIZE:    int   = 6    # lots per leg (HP-constrained)
 
    # ── EMA / volatility parameters ─────────────────────────────────────────
    _EMA_ALPHA:     float = 0.08
    # Blend ratio: 70% EMA + 30% regime midpoint
    # Prevents EMA from drifting too far from the known bounded regime
    _PRIOR_BLEND:   float = 0.10
    _VOL_WINDOW:    int   = 80
    # Default annualised vol: 1.2% daily × √252 ≈ 0.19
    # Derived from the ~125-tick band on a 5250 underlying (±2σ ≈ ±1.2%)
    _SIGMA_DEFAULT: float = 0.19
 
    # ── Time / TTE parameters ───────────────────────────────────────────────
    _ROUND_TTE:     float = 3.0        # game-days at start of this round
    _TICKS_PER_DAY: int   = 1_000_000  # simulation ticks per round
    # Liquidate options when TTE falls below this (ticks ≈ 0.1 game-days)
    _OPT_CLOSE_TTE: float = 0.5
 
    # ── Option edge thresholds ──────────────────────────────────────────────
    # Minimum mispricing (ticks) required before trading vs BS fair value
    _OPT_EDGE_FRAC: float = 0.12   # 12% of theoretical
    _OPT_EDGE_MIN:  float = 4.0    # absolute minimum (ticks)
 
    def run(
        self, state: TradingState
    ) -> tuple[dict[Symbol, list[Order]], int, str]:
 
        result: Dict[str, List[Order]] = {}
        conversions = 0
 
        # ── 1. Restore persisted state ──────────────────────────────────────
        td: Dict = {}
        if state.traderData:
            try:
                td = jsonpickle.decode(state.traderData)
            except Exception:
                td = {}
 
        td.setdefault("vev_prices", [])
        td.setdefault("hp_prices",  [])
        td.setdefault("spreads",    [])
        td.setdefault("vev_ema",    None)
        td.setdefault("hp_ema",     None)
 
        # ── 2. Time-to-expiry (decreases from 5.0 to ≈4.0 across Round 3) ──
        tte: float = max(0.05, self._ROUND_TTE - state.timestamp / self._TICKS_PER_DAY)
 
        # ── 3. Mid prices ───────────────────────────────────────────────────
        vev_od: Optional[OrderDepth] = state.order_depths.get("VELVETFRUIT_EXTRACT")
        hp_od:  Optional[OrderDepth] = state.order_depths.get("HYDROGEL_PACK")
        vev_mid: Optional[float] = self._mid(vev_od)
        hp_mid:  Optional[float] = self._mid(hp_od)
 
        # ── 4. Update EMAs + rolling histories ─────────────────────────────
        a = self._EMA_ALPHA
        if vev_mid is not None:
            prev = td["vev_ema"]
            td["vev_ema"] = (
                vev_mid if prev is None else a * vev_mid + (1.0 - a) * prev
            )
            td["vev_prices"].append(vev_mid)
            td["vev_prices"] = td["vev_prices"][-self._VOL_WINDOW:]
 
        if hp_mid is not None:
            prev = td["hp_ema"]
            td["hp_ema"] = (
                hp_mid if prev is None else a * hp_mid + (1.0 - a) * prev
            )
            td["hp_prices"].append(hp_mid)
            td["hp_prices"] = td["hp_prices"][-self._VOL_WINDOW:]
 
        if vev_mid is not None and hp_mid is not None:
            td["spreads"].append(vev_mid - self._PAIRS_RATIO * hp_mid)
            td["spreads"] = td["spreads"][-self._PAIRS_WINDOW:]
        pair_z: Optional[float] = None
        if len(td["spreads"]) >= 10:
            center = statistics.median(td["spreads"])
            mad = statistics.median([abs(x - center) for x in td["spreads"]]) if len(td["spreads"]) > 1 else 0.0
            scale = mad if mad > 1e-6 else (statistics.stdev(td["spreads"]) if len(td["spreads"]) > 1 else 1.0)
            if scale > 1e-6:
                pair_z = 0.6745 * (td["spreads"][-1] - center) / scale
 
        # ── 5. Fair values (EMA blended with known regime midpoint) ─────────
        # Using a prior blend avoids the EMA drifting during low-liquidity
        # periods and anchors it to the stationary mean known from research.
        p = self._PRIOR_BLEND
        fair_vev: float = (
            (1 - p) * float(td["vev_ema"]) + p * self._VEV_REGIME_MID
            if td["vev_ema"] is not None
            else self._VEV_REGIME_MID
        )
        fair_hp: float
        if vev_mid is not None:
            spread_anchor = statistics.median(td["spreads"]) if td["spreads"] else self._SPREAD_EQ
            fair_hp = 2.0 * (fair_vev - spread_anchor)
        else:
            fair_hp = (
                (1 - p) * float(td["hp_ema"]) + p * self._HP_REGIME_MID
                if td["hp_ema"] is not None
                else self._HP_REGIME_MID
            )
 
        # ── 6. Realised-vol estimate for Black-Scholes ──────────────────────
        sigma: float = self._estimate_sigma(td["vev_prices"])
 
        # ── 7a. VEV: aggressive mean reversion + passive market making ──────
        if vev_od is not None:
            if pair_z is None or abs(pair_z) < 1.25:
                result["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                    state=state,
                    sym="VELVETFRUIT_EXTRACT",
                    od=vev_od,
                    fair=fair_vev,
                    max_pos=self._VEV_MAX_POS,
                    mm_size=self._VEV_MM_SIZE,
                    passive_only=False,
                )
            else:
                # During a strong relative-value signal, let the pair overlay
                # lead so the spot layer doesn't fight it.
                result["VELVETFRUIT_EXTRACT"] = []
 
        # ── 7b. HP: passive-only (30-lot/tick illiquidity constraint) ───────
        if hp_od is not None:
            result["HYDROGEL_PACK"] = self._trade_spot(
                state=state,
                sym="HYDROGEL_PACK",
                od=hp_od,
                fair=fair_hp,
                max_pos=self._HP_MAX_POS,
                mm_size=self._HP_MM_SIZE,
                passive_only=True,
            )
 
        # ── 7c. Pairs overlay (added on top of individual spot positions) ───
        if (
            len(td["spreads"]) >= 20
            and vev_od is not None
            and hp_od is not None
            and vev_mid is not None
            and hp_mid is not None
        ):
            self._pairs_overlay(state, result, vev_od, hp_od, td["spreads"])
 
        # ── 7d. Options ──────────────────────────────────────────────────────
        if vev_mid is not None:
            for sym, strike in self.STRIKES.items():
                od = state.order_depths.get(sym)
                if od is None:
                    continue
                opt_orders = self._trade_option(
                    state=state,
                    sym=sym,
                    strike=strike,
                    S=vev_mid,
                    tte=tte,
                    sigma=sigma,
                    od=od,
                )
                if opt_orders:
                    result[sym] = opt_orders
 
        # ── 8. Safety: clip to hard exchange limits ─────────────────────────
        result = self._enforce_limits(result, state.position)
 
        # ── 9. Prune empty lists ─────────────────────────────────────────────
        result = {k: v for k, v in result.items() if v}
 
        logger.flush(state, result, conversions, jsonpickle.encode(td))
        return result, conversions, jsonpickle.encode(td)
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Spot: mean-reversion market making
    # ─────────────────────────────────────────────────────────────────────────
 
    def _trade_spot(
        self,
        state: TradingState,
        sym: str,
        od: OrderDepth,
        fair: float,
        max_pos: int,
        mm_size: int,
        passive_only: bool,
    ) -> List[Order]:
        """
        Two-mode spot trader:
          Aggressive: cross the spread when the opposing quote is ≥ AGGR_TICKS
                      away from fair value (only if passive_only=False).
          Passive:    post limit orders at fair ± PASSIVE_OFFSET to capture the
                      spread as a market maker.
        """
        orders: List[Order] = []
        pos     = state.position.get(sym, 0)
        # Remaining capacity on each side before hitting our soft cap
        buy_cap  = max(0, max_pos - pos)
        sell_cap = max(0, max_pos + pos)
 
        # Inventory skew discourages leaning too hard into one side.
        fair = fair - 0.15 * pos
        fair_r = round(fair)
        aggr   = self._SPOT_AGGR_TICKS
 
        # ── Aggressive takes ─────────────────────────────────────────────────
        if not passive_only:
            # Buy anything offered ≤ fair − aggr  (clearly underpriced)
            if od.sell_orders and buy_cap > 0:
                for px in sorted(od.sell_orders):
                    if px <= fair_r - aggr:
                        vol = min(-od.sell_orders[px], buy_cap)
                        if vol > 0:
                            orders.append(Order(sym, px, vol))
                            buy_cap -= vol
                    else:
                        break
 
            # Sell into any bid ≥ fair + aggr  (clearly overpriced)
            if od.buy_orders and sell_cap > 0:
                for px in sorted(od.buy_orders, reverse=True):
                    if px >= fair_r + aggr:
                        vol = min(od.buy_orders[px], sell_cap)
                        if vol > 0:
                            orders.append(Order(sym, px, -vol))
                            sell_cap -= vol
                    else:
                        break
 
        # ── Passive quotes (market making) ───────────────────────────────────
        off = self._SPOT_PASSIVE_OFFSET
        bid_px = fair_r - off
        ask_px = fair_r + off
 
        # Sanity: never cross our own quotes
        if bid_px >= ask_px:
            bid_px = fair_r - 1
            ask_px = fair_r + 1
 
        if buy_cap > 0:
            orders.append(Order(sym, bid_px, min(buy_cap, mm_size)))
        if sell_cap > 0:
            orders.append(Order(sym, ask_px, -min(sell_cap, mm_size)))
 
        return orders
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Pairs overlay: statistical arbitrage on VEV / HP spread
    # ─────────────────────────────────────────────────────────────────────────
 
    def _pairs_overlay(
        self,
        state:   TradingState,
        result:  Dict[str, List[Order]],
        vev_od:  OrderDepth,
        hp_od:   OrderDepth,
        spreads: List[float],
    ) -> None:
        """
        Spread = VEV_mid − 0.5 × HP_mid  (equilibrium ≈ 247.5).
 
        Enter long spread (long VEV, short HP) when z-score < −ENTER_Z.
        Enter short spread (short VEV, long HP) when z-score >  ENTER_Z.
        Unwind when |z| < EXIT_Z.
 
        Sizing is deliberately small (_PAIRS_SIZE = 10) because:
          • HP has a 30-lot/tick volume ceiling, so the short leg is illiquid.
          • We already hold spot positions from the mean-reversion layer.
        """
        if len(spreads) < 10:
            return
 
        mean_sp = sum(spreads) / len(spreads)
        std_sp  = statistics.stdev(spreads) if len(spreads) > 1 else 1.0
        if std_sp < 1e-6:
            return
 
        z    = (spreads[-1] - mean_sp) / std_sp
        size = self._PAIRS_SIZE
 
        vev_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        hp_pos  = state.position.get("HYDROGEL_PACK",       0)
 
        if z > self._PAIRS_ENTER_Z:
            # Spread too wide → short VEV (overpriced), long HP (underpriced)
            sz = min(
                size,
                max(0, self._VEV_MAX_POS + vev_pos),  # sell capacity
                max(0, self._HP_MAX_POS  - hp_pos),   # buy capacity
            )
            if sz > 0:
                if vev_od.buy_orders:
                    result.setdefault("VELVETFRUIT_EXTRACT", []).append(
                        Order("VELVETFRUIT_EXTRACT", max(vev_od.buy_orders), -sz)
                    )
                if hp_od.sell_orders:
                    result.setdefault("HYDROGEL_PACK", []).append(
                        Order("HYDROGEL_PACK", min(hp_od.sell_orders), sz)
                    )
 
        elif z < -self._PAIRS_ENTER_Z:
            # Spread too narrow → long VEV (underpriced), short HP (overpriced)
            sz = min(
                size,
                max(0, self._VEV_MAX_POS - vev_pos),  # buy capacity
                max(0, self._HP_MAX_POS  + hp_pos),   # sell capacity
            )
            if sz > 0:
                if vev_od.sell_orders:
                    result.setdefault("VELVETFRUIT_EXTRACT", []).append(
                        Order("VELVETFRUIT_EXTRACT", min(vev_od.sell_orders), sz)
                    )
                if hp_od.buy_orders:
                    result.setdefault("HYDROGEL_PACK", []).append(
                        Order("HYDROGEL_PACK", max(hp_od.buy_orders), -sz)
                    )
 
        elif abs(z) < self._PAIRS_EXIT_Z:
            # Mean-reversion achieved → unwind residual directional exposure
            unwind = min(size, abs(vev_pos))
            if vev_pos > 5 and vev_od.buy_orders and unwind > 0:
                result.setdefault("VELVETFRUIT_EXTRACT", []).append(
                    Order("VELVETFRUIT_EXTRACT", max(vev_od.buy_orders), -unwind)
                )
            elif vev_pos < -5 and vev_od.sell_orders and unwind > 0:
                result.setdefault("VELVETFRUIT_EXTRACT", []).append(
                    Order("VELVETFRUIT_EXTRACT", min(vev_od.sell_orders), unwind)
                )
 
            unwind = min(size, abs(hp_pos))
            if hp_pos > 5 and hp_od.buy_orders and unwind > 0:
                result.setdefault("HYDROGEL_PACK", []).append(
                    Order("HYDROGEL_PACK", max(hp_od.buy_orders), -unwind)
                )
            elif hp_pos < -5 and hp_od.sell_orders and unwind > 0:
                result.setdefault("HYDROGEL_PACK", []).append(
                    Order("HYDROGEL_PACK", min(hp_od.sell_orders), unwind)
                )
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Options: Black-Scholes vs. market
    # ─────────────────────────────────────────────────────────────────────────
 
    def _trade_option(
        self,
        state:  TradingState,
        sym:    str,
        strike: int,
        S:      float,
        tte:    float,
        sigma:  float,
        od:     OrderDepth,
    ) -> List[Order]:
        """
        Strategy derived from research:
 
        Deep OTM (6000, 6500): skip — zero edge, dead strikes.
 
        Near expiry (TTE < 0.5 day): cross the spread to close any open
            position and prevent illiquid end-of-round liquidation at a
            potentially unfavourable hidden fair value.
 
        Deep ITM (≤ 5000, δ ≈ 1): trade only when market price deviates
            from intrinsic value by > 4 ticks (bid-ask noise threshold).
 
        Near ATM / OTM (5100–5500): compare market bid/ask to BS theoretical.
            Sell when market ask > theo + threshold  (overpriced vol premium).
            Buy  when market bid < theo − threshold  (underpriced).
            Post passive MM quotes around theo as a liquidity provider.
 
        The primary directional edge: the market prices OTM calls using
        log-normal assumptions while the underlying is mean-reverting and
        bounded, so OTM calls (especially 5300/5400) carry a structural
        volatility premium that can be harvested by selling them.
        """
        orders: List[Order] = []
 
        # Dead strikes: skip entirely
        if sym in ("VEV_6000", "VEV_6500"):
            return orders
 
        pos = state.position.get(sym, 0)
 
        # ── Near expiry: close position at any price ─────────────────────────
        if tte < self._OPT_CLOSE_TTE:
            if pos > 0 and od.buy_orders:
                orders.append(Order(sym, max(od.buy_orders), -pos))
            elif pos < 0 and od.sell_orders:
                orders.append(Order(sym, min(od.sell_orders), -pos))
            return orders
 
        # ── Risk caps based on moneyness ─────────────────────────────────────
        max_pos  = self._OPT_ITM_MAX if strike <= 5_000 else self._OPT_NEAR_MAX
        buy_cap  = max(0, max_pos - pos)
        sell_cap = max(0, max_pos + pos)
 
        # ── Deep ITM: compare vs. intrinsic (δ ≈ 1, time value negligible) ──
        if strike <= 5_000:
            intrinsic = max(0.0, S - float(strike))
            tol = 4.0   # ticks of allowable noise before we act
 
            if od.sell_orders and buy_cap > 0:
                best_ask = min(od.sell_orders)
                if best_ask < intrinsic - tol:
                    vol = min(-od.sell_orders[best_ask], buy_cap, 20)
                    if vol > 0:
                        orders.append(Order(sym, best_ask, vol))
 
            if od.buy_orders and sell_cap > 0:
                best_bid = max(od.buy_orders)
                if best_bid > intrinsic + tol:
                    vol = min(od.buy_orders[best_bid], sell_cap, 20)
                    if vol > 0:
                        orders.append(Order(sym, best_bid, -vol))
 
            return orders
 
        # ── Near ATM / OTM: Black-Scholes mispricing ─────────────────────────
        theo = _bs_call(S, float(strike), tte, sigma)
 
        # Minimum required edge before trading (wider threshold = fewer noise
        # trades, better Sharpe by avoiding over-trading near fair value)
        threshold = max(self._OPT_EDGE_MIN, theo * self._OPT_EDGE_FRAC)
 
        # Aggressive: take when clearly mispriced
        if od.sell_orders and buy_cap > 0:
            best_ask = min(od.sell_orders)
            if best_ask < theo - threshold:
                vol = min(-od.sell_orders[best_ask], buy_cap, 10)
                if vol > 0:
                    orders.append(Order(sym, best_ask, vol))
                    buy_cap -= vol
 
        if od.buy_orders and sell_cap > 0:
            best_bid = max(od.buy_orders)
            if best_bid > theo + threshold:
                vol = min(od.buy_orders[best_bid], sell_cap, 10)
                if vol > 0:
                    orders.append(Order(sym, best_bid, -vol))
                    sell_cap -= vol
 
        # Passive market making around BS fair value
        # Only post when the option has meaningful value (avoid junk fills)
        if theo >= 2.0:
            mm_bid = math.floor(theo - 1)
            mm_ask = math.ceil(theo + 1)
            if mm_bid >= 1 and buy_cap > 0:
                orders.append(Order(sym, mm_bid, min(buy_cap, 5)))
            if mm_ask >= 1 and sell_cap > 0:
                orders.append(Order(sym, mm_ask, -min(sell_cap, 5)))
 
        return orders
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Risk: enforce hard exchange position limits
    # ─────────────────────────────────────────────────────────────────────────
 
    def _enforce_limits(
        self,
        result:   Dict[str, List[Order]],
        position: Dict[str, int],
    ) -> Dict[str, List[Order]]:
        """
        Safety net: if the total buy (sell) quantity across all orders for
        a symbol would breach the exchange position limit, clip orders from
        the back of the list until we are within the limit.
 
        This prevents exchange-side rejection of the ENTIRE order batch.
        Under normal operation this should not trigger, but it protects
        against edge cases where the pairs overlay and spot strategy both
        send orders in the same direction.
        """
        clean: Dict[str, List[Order]] = {}
 
        for sym, orders in result.items():
            pos   = position.get(sym, 0)
            limit = self.LIMITS.get(sym, 300)
 
            remaining_buy  = max(0, limit - pos)
            remaining_sell = max(0, limit + pos)
 
            clipped: List[Order] = []
            for o in orders:
                if o.quantity > 0:
                    if remaining_buy <= 0:
                        continue
                    qty = min(o.quantity, remaining_buy)
                    clipped.append(Order(o.symbol, o.price, qty))
                    remaining_buy -= qty
                elif o.quantity < 0:
                    if remaining_sell <= 0:
                        continue
                    qty = min(-o.quantity, remaining_sell)
                    clipped.append(Order(o.symbol, o.price, -qty))
                    remaining_sell -= qty
 
            clean[sym] = clipped
 
        return clean
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Volatility: realised vol from rolling tick prices
    # ─────────────────────────────────────────────────────────────────────────
 
    def _estimate_sigma(self, prices: List[float]) -> float:
        """
        Estimate annualised σ from stored mid prices.
 
        Each stored price corresponds to one simulation tick (100 ms).
        There are ~10 000 ticks per game-day, so:
            σ_daily   = σ_per_tick × √10 000
            σ_annual  = σ_daily   × √252
 
        Falls back to _SIGMA_DEFAULT if insufficient history or arithmetic
        error (e.g. zero prices in early warm-up ticks).
        """
        if len(prices) < 5:
            return self._SIGMA_DEFAULT
        try:
            log_rets = [
                math.log(prices[i] / prices[i - 1])
                for i in range(1, len(prices))
                if prices[i - 1] > 0.0 and prices[i] > 0.0
            ]
            if len(log_rets) < 4:
                return self._SIGMA_DEFAULT
 
            std_per_tick = statistics.stdev(log_rets)
            sigma_annual = std_per_tick * math.sqrt(10_000 * 252)
            return max(0.05, min(0.80, sigma_annual))
        except Exception:
            return self._SIGMA_DEFAULT
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Helper: mid-price from an OrderDepth
    # ─────────────────────────────────────────────────────────────────────────
 
    @staticmethod
    def _mid(od: Optional[OrderDepth]) -> Optional[float]:
        if od is None:
            return None
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        if od.buy_orders:
            return float(max(od.buy_orders))
        if od.sell_orders:
            return float(min(od.sell_orders))
        return None

