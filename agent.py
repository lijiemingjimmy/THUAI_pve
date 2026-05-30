from __future__ import annotations

from collections import deque
from enum import IntEnum
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from RLInterfaces import BaseAgent


class Action(IntEnum):
    WAIT = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    BUY = 5
    SELL_0 = 6
    SELL_1 = 7
    SELL_2 = 8
    SELL_3 = 9
    SELL_4 = 10
    HARVEST = 11
    DEPOSIT = 12
    PRODUCE_0 = 13
    PRODUCE_1 = 14
    PRODUCE_2 = 15
    PRODUCE_3 = 16
    PRODUCE_4 = 17
    LOAD = 18
    OCCUPY = 19
    TECH_0 = 20
    TECH_1 = 21
    TECH_2 = 22
    TECH_3 = 23
    TECH_4 = 24
    TECH_5 = 25
    TECH_6 = 26
    TECH_7 = 27


MOVE_DELTAS = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}

PRODUCT_DEFS = {
    0: {"raw_cost": 5, "produce_time": 5.0},
    1: {"raw_cost": 3, "produce_time": 4.0},
    2: {"raw_cost": 1, "produce_time": 2.0},
    3: {"raw_cost": 4, "produce_time": 6.0},
    4: {"raw_cost": 2, "produce_time": 1.0},
}


class Agent(BaseAgent):
    """Rule-only PvE policy.

    This is intentionally env-aware: official_evaluator passes env into Agent(env),
    and for a rule bot the exact board/market/factory state is much better signal
    than reverse-engineering everything from the 82-float observation vector.
    """

    TECH_PRIORITY = [
        Action.TECH_0,  # cost_reduction: directly improves arbitrage margin
        Action.TECH_2,  # marketing: directly improves sell revenue
        Action.TECH_1,  # efficiency unlocks path_optimization
        Action.TECH_5,  # path_optimization cuts movement wait ticks
        Action.TECH_7,  # compute_expansion only matters if bought early
        Action.TECH_4,  # production-only fallback
        Action.TECH_3,  # carry capacity is weak for one-item arbitrage
        Action.TECH_6,  # market_analysis is low value for env-aware policy
    ]

    MOVE_ACTIONS = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]

    def get_action(self, observation: np.ndarray) -> int:
        mask = self.env.action_masks()
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return int(Action.WAIT)

        if self.env.unit.busy_ticks > 0:
            return int(Action.WAIT)

        sell_action = self._best_sell_action(mask)
        if sell_action is not None:
            return sell_action

        if self._should_batch_buy_arbitrage(mask):
            return int(Action.BUY)

        if self._carried_products() > 0:
            move = self._move_to_best_sell_market(mask)
            if move is not None:
                return move

        if self._should_buy_arbitrage(mask):
            return int(Action.BUY)

        move = self._move_to_best_buy_market(mask)
        if move is not None:
            return move

        return int(valid[0])

    def train(self, total_timesteps: int, **kwargs):
        return {"total_timesteps": 0, "policy": "rule"}

    def save(self, path: str):
        Path(path).write_text("rule_policy\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | None, env):
        return cls(env)

    def _best_sell_action(self, mask: np.ndarray) -> Optional[int]:
        mkt = self._adjacent_market()
        if mkt is None:
            return None
        best: tuple[float, int] | None = None
        for pid, qty in self.env.unit.all_products().items():
            act = int(Action.SELL_0) + int(pid)
            if not mask[act]:
                continue
            price = mkt.get_price(pid, self.env._price_multiplier())
            value = price * qty
            if best is None or value > best[0]:
                best = (value, act)
        return None if best is None else int(best[1])

    def _best_tech(self, mask: np.ndarray) -> Optional[int]:
        # Do not spend the first compute on market_analysis; it is transient and
        # this policy can inspect env markets directly.
        for action in self.TECH_PRIORITY:
            if mask[action]:
                return int(action)
        return None

    def _best_produce_action(self, mask: np.ndarray) -> Optional[int]:
        best: tuple[float, int] | None = None
        sell_mult = self.env._price_multiplier()
        for pid, pdef in PRODUCT_DEFS.items():
            action = int(Action.PRODUCE_0) + pid
            if not mask[action]:
                continue
            best_price = max(m.get_price(pid, sell_mult) for m in self.env.markets)
            raw_cost = max(1.0, float(pdef["raw_cost"]))
            produce_time = max(0.25, float(pdef["produce_time"]) * self.env.factory.time_multiplier)
            # Raw is the bottleneck, but time matters enough to avoid over-queuing slow goods.
            score = best_price / raw_cost / (1.0 + 0.08 * produce_time)
            if best is None or score > best[0]:
                best = (score, action)
        return None if best is None else int(best[1])

    def _should_buy_arbitrage(self, mask: np.ndarray) -> bool:
        return self._arbitrage_buy_score(mask) is not None

    def _should_batch_buy_arbitrage(self, mask: np.ndarray) -> bool:
        score = self._arbitrage_buy_score(mask)
        if score is None:
            return False
        mkt = self._adjacent_market()
        if mkt is None:
            return False
        # If already carrying goods from another market, move to sell first.
        # If goods are from this same market, keep filling the bag to amortize travel.
        for origins in self.env.unit.prod_origin.values():
            for origin_market, qty in origins.items():
                if qty > 0 and origin_market != mkt.id:
                    return False
        carried = self._carried_products()
        if carried <= 0:
            return True
        # Stop a little before the end; selling a full bag still needs travel plus one sell per product.
        remaining_steps = self.env.cfg.max_steps - self.env._step
        if remaining_steps < 30:
            return False
        return self.env.unit.free_capacity >= 1

    def _arbitrage_buy_score(self, mask: np.ndarray) -> Optional[float]:
        if not mask[Action.BUY] or self.env.unit.free_capacity < 1:
            return None
        mkt = self._adjacent_market()
        if mkt is None:
            return None
        pid, cost = self.env._best_buyable(mkt)
        if pid is None or cost is None:
            return None
        best_other = max(
            om.get_price(pid, self.env._price_multiplier())
            for om in self.env.markets
            if om.id != mkt.id
        )
        spread = best_other - cost
        if spread < max(2.0, cost * 0.12):
            return None
        return spread

    def _should_open_compute_center(self, mask: np.ndarray) -> bool:
        return False
        if self.env.time > self.env.cfg.max_game_time * 0.45:
            return False
        if self.env.unit.raw_inv > 0 or self._carried_products() > 0:
            return False
        if any(cc.is_open for cc in self.env.board.compute_centers):
            return False
        # Open one early center so durability/efficiency/multiline become reachable.
        return bool(mask[Action.OCCUPY]) or self._nearest_closed_compute_cell() is not None

    def _move_to_best_resource(self, mask: np.ndarray) -> Optional[int]:
        resources = [rp for rp in self.env.board.resource_points if not rp.depleted and rp.stock > 0]
        if not resources:
            return None
        ux, uy = self.env.unit.x, self.env.unit.y
        resources.sort(key=lambda rp: (-rp.stock, abs(rp.x - ux) + abs(rp.y - uy)))
        for rp in resources:
            move = self._move_toward_any(mask, self._cells_within(rp.x, rp.y, 2))
            if move is not None:
                return move
        return None

    def _move_to_best_sell_market(self, mask: np.ndarray) -> Optional[int]:
        products = self.env.unit.all_products()
        if not products:
            return None
        sell_mult = self.env._price_multiplier()
        ranked = []
        for mkt in self.env.markets:
            value = 0.0
            for pid, qty in products.items():
                blocked = self.env.unit.origin_qty(pid, mkt.id)
                sell_qty = max(0.0, qty - blocked)
                value += sell_qty * mkt.get_price(pid, sell_mult)
            if value > 0:
                ranked.append((value, mkt))
        scored = []
        start = (self.env.unit.x, self.env.unit.y)
        for value, mkt in ranked:
            cells = self._market_cells(mkt)
            dist = self._distance_to_any_from(start, cells)
            if dist is None:
                continue
            scored.append((value / max(1.0, float(dist)), value, mkt))
        scored.sort(key=lambda item: item[0], reverse=True)
        for _score, _value, mkt in scored:
            move = self._move_toward_any(mask, self._market_cells(mkt))
            if move is not None:
                return move
        return None

    def _move_to_best_buy_market(self, mask: np.ndarray) -> Optional[int]:
        ranked = []
        start = (self.env.unit.x, self.env.unit.y)
        for mkt in self.env.markets:
            pid, cost = self.env._best_buyable(mkt)
            if pid is None or cost is None:
                continue
            sell_options = []
            for om in self.env.markets:
                if om.id == mkt.id:
                    continue
                spread = om.get_price(pid, self.env._price_multiplier()) - cost
                sell_options.append((spread, om))
            if not sell_options:
                continue
            spread, sell_mkt = max(sell_options, key=lambda item: item[0])
            if spread < 2.0:
                continue
            buy_cells = self._market_cells(mkt)
            sell_cells = self._market_cells(sell_mkt)
            dist_to_buy = self._distance_to_any_from(start, buy_cells)
            if dist_to_buy is None:
                continue
            dists = [self._distance_to_any_from(cell, sell_cells) for cell in buy_cells]
            dists = [dist for dist in dists if dist is not None]
            if not dists:
                continue
            cycle_dist = max(1.0, float(dist_to_buy + min(dists)))
            ranked.append((spread / cycle_dist, spread, mkt))
        ranked.sort(key=lambda item: item[0], reverse=True)
        for _score, _spread, mkt in ranked:
            move = self._move_toward_any(mask, self._market_cells(mkt))
            if move is not None:
                return move
        return None

    def _move_to_factory(self, mask: np.ndarray) -> Optional[int]:
        return self._move_toward_any(mask, [(self.env.cfg.factory_x, self.env.cfg.factory_y)])

    def _move_to_nearest_compute(self, mask: np.ndarray) -> Optional[int]:
        cell = self._nearest_closed_compute_cell()
        if cell is None:
            return None
        return self._move_toward_any(mask, [cell])

    def _nearest_closed_compute_cell(self) -> Optional[tuple[int, int]]:
        ux, uy = self.env.unit.x, self.env.unit.y
        candidates = []
        for cc in self.env.board.compute_centers:
            if cc.is_open:
                continue
            for cell in self._cells_within(cc.x, cc.y, 1):
                if self.env.board.is_passable(*cell):
                    candidates.append((abs(cell[0] - ux) + abs(cell[1] - uy), cell))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _move_toward_any(self, mask: np.ndarray, goals: Iterable[tuple[int, int]]) -> Optional[int]:
        goal_set = {g for g in goals if self.env.board.is_passable(*g)}
        if not goal_set:
            return None
        start = (self.env.unit.x, self.env.unit.y)
        if start in goal_set:
            return None

        first = self._bfs_first_step(start, lambda cell: cell in goal_set)
        if first is not None and mask[first]:
            return int(first)

        # If the exact BFS first step became masked by a transient issue, use any legal move reducing distance.
        best: tuple[int, int] | None = None
        for action, (dx, dy) in MOVE_DELTAS.items():
            if not mask[action]:
                continue
            nx, ny = start[0] + dx, start[1] + dy
            dist = min(abs(nx - gx) + abs(ny - gy) for gx, gy in goal_set)
            if best is None or dist < best[0]:
                best = (dist, int(action))
        return None if best is None else best[1]

    def _bfs_first_step(
        self,
        start: tuple[int, int],
        is_goal: Callable[[tuple[int, int]], bool],
    ) -> Optional[int]:
        queue = deque([start])
        parent: dict[tuple[int, int], tuple[tuple[int, int], Action] | None] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell != start and is_goal(cell):
                cur = cell
                prev = parent[cur]
                while prev is not None and prev[0] != start:
                    cur = prev[0]
                    prev = parent[cur]
                return None if prev is None else int(prev[1])
            for action, (dx, dy) in MOVE_DELTAS.items():
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in parent or not self.env.board.is_passable(*nxt):
                    continue
                parent[nxt] = (cell, action)
                queue.append(nxt)
        return None

    def _market_cells(self, mkt) -> list[tuple[int, int]]:
        # game_env sells/buys through board.nearest_market(), which returns the
        # first market within range. Overlapping market ranges can otherwise trap
        # us on a cell that is geometrically adjacent to the target but resolves
        # to another market and makes SELL invalid.
        cells = []
        for cell in self._cells_within(mkt.x, mkt.y, 1):
            if not self.env.board.is_passable(*cell):
                continue
            active = self.env.board.nearest_market(*cell)
            if active == (mkt.x, mkt.y):
                cells.append(cell)
        return cells

    def _cells_within(self, x: int, y: int, radius: int) -> list[tuple[int, int]]:
        cells = []
        for i in range(self.env.board.H):
            for j in range(self.env.board.W):
                if abs(i - x) + abs(j - y) <= radius:
                    cells.append((i, j))
        return cells

    def _distance_to_any_from(self, start: tuple[int, int], goals: Iterable[tuple[int, int]]) -> Optional[int]:
        goal_set = {g for g in goals if self.env.board.is_passable(*g)}
        if not goal_set:
            return None
        if start in goal_set:
            return 0
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            cell, dist = queue.popleft()
            for _action, (dx, dy) in MOVE_DELTAS.items():
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in seen or not self.env.board.is_passable(*nxt):
                    continue
                if nxt in goal_set:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def _adjacent_market(self):
        pos = self.env.board.nearest_market(self.env.unit.x, self.env.unit.y)
        if pos is None:
            return None
        return self.env.market_at(*pos)

    def _carried_products(self) -> float:
        return float(sum(self.env.unit.prod_inv.values()))
