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
 
    # ── Regime priors ────────────────────────────────────────────────────────
    _VEV_REGIME_MID: float = 5_237.5
 
    # ── FIX: HYDROGEL_PACK is DISABLED ──────────────────────────────────────
    # Root-cause: the passive MM with a laggy EMA fair value gets adversely
    # selected on every oscillation of HP's ~200-tick range.  With a 20-lot
    # soft cap, each reversal costs ≈ 20 × 100 = 2 000 ticks of P&L, and with
    # hundreds of reversals per day the loss accumulates to -235 000/day.
    # Additionally HP almost certainly has a conversion mechanism whose fees
    # shift the true fair value away from the order-book mid, making the EMA
    # prior systematically wrong.  Until the conversion observations are
    # incorporated into the fair-value model, trading HP destroys edge.
    _TRADE_HP: bool = False   # ← set True only once conversion cost is modelled
 
    # ── VEV soft cap (65% of exchange limit = 130 lots) ─────────────────────
    _VEV_MAX_POS:  int = 130
 
    # ── Options position caps ────────────────────────────────────────────────
    _OPT_NEAR_MAX: int = 50   # strikes 5100–5500
    _OPT_ITM_MAX:  int = 30   # strikes ≤ 5000  (reduced from 40, tighter risk)
    # Deep OTM 6000/6500 → skipped entirely (zero edge, dead strikes)
 
    # ── Market-making parameters ─────────────────────────────────────────────
    # FIX: Increased aggression threshold from 3 → 5 ticks to reduce noise
    # trades and adverse selection on VEV.
    _SPOT_AGGR_TICKS:    int   = 5
    # FIX: Widened passive offset from 1 → 2 so the bot earns a better spread
    # and is harder to pick off by smarter participants.
    _SPOT_PASSIVE_OFFSET: int  = 2
    _VEV_MM_SIZE: int = 12
 
    # ── Pairs-trading parameters ─────────────────────────────────────────────
    # FIX: Pairs overlay only runs on VEV now (HP leg removed).
    # Raised enter z from 2.0 → 2.5 to reduce false entries.
    _PAIRS_RATIO:   float = 0.5
    _PAIRS_WINDOW:  int   = 50
    _PAIRS_ENTER_Z: float = 2.5
    _PAIRS_EXIT_Z:  float = 0.5
    _PAIRS_VEV_SIZE: int  = 8   # VEV-only pairs leg size
 
    # ── EMA / volatility parameters ──────────────────────────────────────────
    # FIX: Increased alpha from 0.08 → 0.15 so EMA tracks price more quickly,
    # reducing the lag that caused adverse selection.
    _EMA_ALPHA: float = 0.15
    # FIX: Reduced prior blend from 0.30 → 0.10.  The regime mid-point is only
    # a weak anchor; trusting the live EMA more improves fair-value accuracy.
    _PRIOR_BLEND: float = 0.10
    _VOL_WINDOW:  int   = 60
    # Default sigma: 1.2% daily × √252 ≈ 0.19 (unchanged, reasonable prior)
    _SIGMA_DEFAULT: float = 0.19
 
    # ── Time / TTE parameters ────────────────────────────────────────────────
    # FIX: _ROUND_TTE was labelled "Round 3" but this is Round 4.
    # Round 4 has 3 trading days. We track which day we're on via traderData
    # and subtract the elapsed days from TOTAL_ROUND_DAYS (= 3).
    # TTE at start of day 1: 3.0 days  (3 full days remain)
    # TTE at start of day 2: 2.0 days
    # TTE at start of day 3: 1.0 day
    # TTE at end   of day 3: ~0 days  (options expire)
    _TOTAL_ROUND_DAYS:  int   = 3
    _TICKS_PER_DAY:     int   = 1_000_000   # ms per day (10 000 ticks × 100 ms)
    # Close options when TTE < 0.2 game-days (≈ 2 000 ticks to expiry)
    _OPT_CLOSE_TTE: float = 0.2
 
    # ── Option edge thresholds ───────────────────────────────────────────────
    # FIX: Raised minimum edge from 3.0 → 4.0 ticks (stricter entry).
    # Raised fraction from 8% → 10% of theoretical (fewer noise fills).
    _OPT_EDGE_FRAC: float = 0.10
    _OPT_EDGE_MIN:  float = 4.0
 
    # ────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────────
 
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
 
    def _estimate_sigma(self, prices: List[float]) -> float:
        """
        Annualised vol from stored mid prices.
        Each stored price = one simulation tick (100 ms).
        σ_daily  = σ_per_tick × √10 000
        σ_annual = σ_daily × √252
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
 
    # ────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ────────────────────────────────────────────────────────────────────────
 
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
        td.setdefault("spreads",    [])
        td.setdefault("vev_ema",    None)
        # Day tracking (persists across ticks WITHIN a day via traderData)
        td.setdefault("current_day",   0)      # 0-based: day 0, 1, 2
        td.setdefault("prev_timestamp", -1)
 
        # ── 2. FIX: Detect new trading day and advance day counter ───────────
        # In Prosperity, timestamps reset to 0 at the start of each new day.
        # A large backward jump in timestamp signals a day rollover.
        prev_ts: int = td["prev_timestamp"]
        if prev_ts > 0 and state.timestamp < prev_ts - 500_000:
            td["current_day"] = td["current_day"] + 1
        td["prev_timestamp"] = state.timestamp
 
        # ── 3. FIX: Time-to-expiry accounting for which day we are on ───────
        # day 0 → TTE starts at 3.0 and decays to 2.0 over the day
        # day 1 → TTE starts at 2.0 and decays to 1.0
        # day 2 → TTE starts at 1.0 and decays to 0.0
        current_day = td["current_day"]
        days_at_start = self._TOTAL_ROUND_DAYS - current_day   # 3, 2, 1
        tte: float = max(
            0.05,
            days_at_start - state.timestamp / self._TICKS_PER_DAY
        )
 
        # ── 4. Mid prices ───────────────────────────────────────────────────
        vev_od: Optional[OrderDepth] = state.order_depths.get("VELVETFRUIT_EXTRACT")
        hp_od:  Optional[OrderDepth] = state.order_depths.get("HYDROGEL_PACK")
        vev_mid: Optional[float] = self._mid(vev_od)
 
        # ── 5. Update EMA + rolling histories ──────────────────────────────
        a = self._EMA_ALPHA
        if vev_mid is not None:
            prev = td["vev_ema"]
            td["vev_ema"] = (
                vev_mid if prev is None else a * vev_mid + (1.0 - a) * prev
            )
            td["vev_prices"].append(vev_mid)
            td["vev_prices"] = td["vev_prices"][-self._VOL_WINDOW:]
 
        # Spread history (VEV only pairs, HP leg removed)
        hp_mid: Optional[float] = self._mid(hp_od)
        if vev_mid is not None and hp_mid is not None:
            td["spreads"].append(vev_mid - self._PAIRS_RATIO * hp_mid)
            td["spreads"] = td["spreads"][-self._PAIRS_WINDOW:]
 
        # ── 6. Fair value for VEV ───────────────────────────────────────────
        # FIX: Reduced prior blend to 0.10 (was 0.30).  The regime anchor
        # drags the estimate toward 5237.5 even when VEV has moved away.
        p = self._PRIOR_BLEND
        fair_vev: float = (
            (1 - p) * float(td["vev_ema"]) + p * self._VEV_REGIME_MID
            if td["vev_ema"] is not None
            else self._VEV_REGIME_MID
        )
 
        # ── 7. Realised-vol estimate for Black-Scholes ──────────────────────
        sigma: float = self._estimate_sigma(td["vev_prices"])
 
        # ── 8a. VEV: aggressive mean reversion + passive market making ──────
        if vev_od is not None:
            result["VELVETFRUIT_EXTRACT"] = self._trade_spot(
                state=state,
                sym="VELVETFRUIT_EXTRACT",
                od=vev_od,
                fair=fair_vev,
                max_pos=self._VEV_MAX_POS,
                mm_size=self._VEV_MM_SIZE,
                passive_only=False,
            )
 
        # ── 8b. FIX: HYDROGEL_PACK trading is DISABLED ──────────────────────
        # Until HP conversion costs are modelled, do not market-make HP.
        # Keeping HP flat saves ~235 000 per day in adverse-selection losses.
        #
        # If you want to re-enable HP in future rounds, do:
        #   result["HYDROGEL_PACK"] = self._trade_spot(...)
        # after incorporating conversionObservations to derive true fair value:
        #   obs = state.observations.conversionObservations.get("HYDROGEL_PACK")
        #   if obs:
        #       import_cost = obs.askPrice + obs.transportFees + obs.importTariff
        #       export_rev  = obs.bidPrice - obs.transportFees - obs.exportTariff
        #       fair_hp = (import_cost + export_rev) / 2.0
        #       ... then market-make with very tight soft cap
        pass  # HP: do nothing
 
        # ── 8c. VEV-only pairs overlay (HP leg removed) ─────────────────────
        # FIX: Pairs overlay used to also trade HP, contributing to HP's loss.
        # Now it only adjusts VEV exposure relative to the spread z-score.
        if (
            len(td["spreads"]) >= self._PAIRS_WINDOW // 2
            and vev_od is not None
            and vev_mid is not None
        ):
            self._vev_pairs_overlay(state, result, vev_od, td["spreads"])
 
        # ── 8d. Options ──────────────────────────────────────────────────────
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
 
        # ── 9. Safety: clip to hard exchange limits ─────────────────────────
        result = self._enforce_limits(result, state.position)
 
        # ── 10. Prune empty lists ────────────────────────────────────────────
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
          Aggressive: cross the spread when opposing quote is ≥ AGGR_TICKS
                      away from fair value (only if passive_only=False).
          Passive:    post limit orders at fair ± PASSIVE_OFFSET to capture
                      the spread as a market maker.
        """
        orders: List[Order] = []
        pos     = state.position.get(sym, 0)
        buy_cap  = max(0, max_pos - pos)
        sell_cap = max(0, max_pos + pos)
 
        fair_r = round(fair)
        aggr   = self._SPOT_AGGR_TICKS
 
        # ── Aggressive takes ─────────────────────────────────────────────────
        if not passive_only:
            if od.sell_orders and buy_cap > 0:
                for px in sorted(od.sell_orders):
                    if px <= fair_r - aggr:
                        vol = min(-od.sell_orders[px], buy_cap)
                        if vol > 0:
                            orders.append(Order(sym, px, vol))
                            buy_cap -= vol
                    else:
                        break
 
            if od.buy_orders and sell_cap > 0:
                for px in sorted(od.buy_orders, reverse=True):
                    if px >= fair_r + aggr:
                        vol = min(od.buy_orders[px], sell_cap)
                        if vol > 0:
                            orders.append(Order(sym, px, -vol))
                            sell_cap -= vol
                    else:
                        break
 
        # ── Passive quotes ───────────────────────────────────────────────────
        off     = self._SPOT_PASSIVE_OFFSET
        bid_px  = fair_r - off
        ask_px  = fair_r + off
 
        # Never cross quotes
        if bid_px >= ask_px:
            bid_px = fair_r - 1
            ask_px = fair_r + 1
 
        if buy_cap > 0:
            orders.append(Order(sym, bid_px, min(buy_cap, mm_size)))
        if sell_cap > 0:
            orders.append(Order(sym, ask_px, -min(sell_cap, mm_size)))
 
        return orders
 
    # ─────────────────────────────────────────────────────────────────────────
    #  VEV-only pairs overlay
    # ─────────────────────────────────────────────────────────────────────────
 
    def _vev_pairs_overlay(
        self,
        state:   TradingState,
        result:  Dict[str, List[Order]],
        vev_od:  OrderDepth,
        spreads: List[float],
    ) -> None:
        """
        FIX: The original pairs overlay traded BOTH VEV and HP legs.
        Trading HP added to HP's already-catastrophic loss.
 
        New design: only adjust VEV exposure based on spread z-score.
        When spread z > +ENTER_Z: VEV is expensive relative to HP → trim VEV.
        When spread z < -ENTER_Z: VEV is cheap relative to HP → add VEV.
        This uses the spread signal without touching the broken HP strategy.
        """
        if len(spreads) < 10:
            return
 
        mean_sp = sum(spreads) / len(spreads)
        std_sp  = statistics.stdev(spreads) if len(spreads) > 1 else 1.0
        if std_sp < 1e-6:
            return
 
        z    = (spreads[-1] - mean_sp) / std_sp
        size = self._PAIRS_VEV_SIZE
 
        vev_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
 
        if z > self._PAIRS_ENTER_Z:
            # VEV overpriced relative to HP → sell VEV
            sell_cap = max(0, self._VEV_MAX_POS + vev_pos)
            sz = min(size, sell_cap)
            if sz > 0 and vev_od.buy_orders:
                result.setdefault("VELVETFRUIT_EXTRACT", []).append(
                    Order("VELVETFRUIT_EXTRACT", max(vev_od.buy_orders), -sz)
                )
 
        elif z < -self._PAIRS_ENTER_Z:
            # VEV underpriced relative to HP → buy VEV
            buy_cap = max(0, self._VEV_MAX_POS - vev_pos)
            sz = min(size, buy_cap)
            if sz > 0 and vev_od.sell_orders:
                result.setdefault("VELVETFRUIT_EXTRACT", []).append(
                    Order("VELVETFRUIT_EXTRACT", min(vev_od.sell_orders), sz)
                )
 
        elif abs(z) < self._PAIRS_EXIT_Z:
            # Mean-reversion: unwind any directional exposure
            unwind = min(size, abs(vev_pos))
            if vev_pos > 5 and vev_od.buy_orders and unwind > 0:
                result.setdefault("VELVETFRUIT_EXTRACT", []).append(
                    Order("VELVETFRUIT_EXTRACT", max(vev_od.buy_orders), -unwind)
                )
            elif vev_pos < -5 and vev_od.sell_orders and unwind > 0:
                result.setdefault("VELVETFRUIT_EXTRACT", []).append(
                    Order("VELVETFRUIT_EXTRACT", min(vev_od.sell_orders), unwind)
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
        FIX summary:
        1. TTE is now correctly computed per-day (passed from run()).
        2. Deep ITM (≤ 5000) position cap reduced from 40 → 30.
        3. Near-expiry threshold raised to 0.2 days (was 0.5) so we exit
           options sooner before illiquid end-of-round settlement.
        4. Passive MM removed: posting MM quotes around BS fair value churned
           fills on both sides without realised edge.  Now ONLY aggressive
           takes when mispricing exceeds threshold.
        5. Per-trade size capped at 10 lots (was 15) for tighter risk.
        6. FIX: Dead strikes (6000, 6500) and deep-ITM with zero market
           presence (4000, 4500) are skipped entirely.
        """
        orders: List[Order] = []
 
        # Skip dead / illiquid strikes
        if sym in ("VEV_6000", "VEV_6500", "VEV_4000", "VEV_4500"):
            return orders
 
        pos = state.position.get(sym, 0)
 
        # ── Near expiry: close position aggressively ─────────────────────────
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
 
        # ── Deep ITM (5000): intrinsic value comparison ──────────────────────
        if strike <= 5_000:
            intrinsic = max(0.0, S - float(strike))
            # FIX: Raised tolerance from 4 → 6 ticks to filter noise
            tol = 6.0
 
            if od.sell_orders and buy_cap > 0:
                best_ask = min(od.sell_orders)
                if best_ask < intrinsic - tol:
                    vol = min(-od.sell_orders[best_ask], buy_cap, 10)
                    if vol > 0:
                        orders.append(Order(sym, best_ask, vol))
 
            if od.buy_orders and sell_cap > 0:
                best_bid = max(od.buy_orders)
                if best_bid > intrinsic + tol:
                    vol = min(od.buy_orders[best_bid], sell_cap, 10)
                    if vol > 0:
                        orders.append(Order(sym, best_bid, -vol))
 
            return orders
 
        # ── Near ATM / OTM (5100–5500): BS mispricing ────────────────────────
        theo = _bs_call(S, float(strike), tte, sigma)
 
        # Skip if theoretical value is negligible (deep OTM, near expiry)
        if theo < 1.0:
            return orders
 
        threshold = max(self._OPT_EDGE_MIN, theo * self._OPT_EDGE_FRAC)
 
        # FIX: Removed passive MM quotes (bid/ask at theo±1).
        # Passive MM churned P&L without consistent edge because:
        #   a) fills on both sides net to zero with 2-tick gross spread, and
        #   b) any adverse selection (faster bots knowing next price) cost > 2.
        # Instead: only AGGRESSIVE takes when clearly mispriced.
 
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
        Safety net: clip orders so total filled quantity can never breach
        the exchange position limit.  Under normal operation this should
        not trigger; it guards against edge cases.
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

