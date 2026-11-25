from __future__ import annotations

from itertools import combinations
from math import hypot

from core.action import Action
from core.depth import Depth
from core.player import Player
from core.seasons import Season
from core.snapshot import JellyfishSnapshot

_WORLD = 1000
_ZONE_RADIUS = 200
_VERSION = 0xA6

_POS_X = 1
_POS_Y = 3
_LAST_AGE = 5
_KNOWN_MASK = 9
_VELOCITIES = 10
_PREVIOUS_ZONES = 18
_SAMPLE_STATES = 19
_CENTERS = 22
_SAMPLES = 37
_ZONE_MEMORY_SIZE = 85
_GRID_BASE = _ZONE_MEMORY_SIZE
_EXTRA_SAMPLES = _GRID_BASE + 20 * 20
_EXTRA_SAMPLE_STATES = _EXTRA_SAMPLES + 3 * 4 * 4
_EXTRA_SAMPLE_MEMORY_SIZE = _EXTRA_SAMPLE_STATES + 3

_FEEDING = 0
_BREEDING = 1
_PREDATOR = 2
_ZONE_FLAGS = (_FEEDING, _BREEDING, _PREDATOR)


def _wrap(value: float) -> float:
    return value % _WORLD


def _delta(a: float, b: float) -> float:
    return (a - b + _WORLD / 2) % _WORLD - _WORLD / 2


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(_delta(a[0], b[0]), _delta(a[1], b[1]))


def _get_u16(memory: bytearray, offset: int) -> int:
    return memory[offset] | memory[offset + 1] << 8


def _put_u16(memory: bytearray, offset: int, value: float) -> None:
    value = round(value) % _WORLD
    memory[offset] = value & 0xFF
    memory[offset + 1] = value >> 8


def _get_i8(memory: bytearray, offset: int) -> int:
    value = memory[offset]
    return value - 256 if value >= 128 else value


def _put_i8(memory: bytearray, offset: int, value: float) -> None:
    memory[offset] = round(value) & 0xFF


class Player6(Player):
    """Navigate between seasonal zones while learning currents and locations.

    The saved position is relative to each jellyfish's birth position. Its
    memory is also the only persistent state used by this policy.
    """

    def act(self, snapshot: JellyfishSnapshot) -> Action:
        target_zone = (
            _FEEDING if snapshot.season in (Season.SPRING, Season.SUMMER) else _BREEDING
        )
        in_target = (
            snapshot.in_feeding_zone
            if target_zone == _FEEDING
            else snapshot.in_breeding_zone
        )

        if len(snapshot.memory) < _PREVIOUS_ZONES + 1:
            depth = self._small_memory_action(snapshot, in_target)
        else:
            depth = self._tracked_action(snapshot, target_zone, in_target)

        eat = snapshot.season == Season.SUMMER and snapshot.in_feeding_zone
        reproduce = (
            snapshot.season == Season.WINTER
            and snapshot.in_breeding_zone
            and snapshot.reproduction_capacity > 0
        )
        return Action(depth=depth, eat=eat, reproduce=reproduce)

    def _small_memory_action(
        self, snapshot: JellyfishSnapshot, in_target: bool
    ) -> Depth:
        slowest = min(Depth, key=lambda depth: depth.speed)
        if in_target:
            return slowest
        if snapshot.in_predator_zone:
            return snapshot.depth

        packed = snapshot.memory[0]
        depth_value, days_left = packed >> 6, packed & 0x3F
        if days_left == 0:
            depth_value = self.rng.randrange(len(Depth))
            days_left = self.rng.randint(10, 30)
        snapshot.memory[0] = (depth_value << 6) | (days_left - 1)
        return Depth(depth_value)

    def _tracked_action(
        self, snapshot: JellyfishSnapshot, target_zone: int, in_target: bool
    ) -> Depth:
        memory = snapshot.memory
        if memory[0] != _VERSION:
            for index in range(len(memory)):
                memory[index] = 0
            memory[0] = _VERSION
            _put_u16(memory, _POS_X, 0)
            _put_u16(memory, _POS_Y, 0)
            memory[_PREVIOUS_ZONES] = self._zone_flags(snapshot)
            self._set_age(memory, snapshot.age_days)
            old_x = old_y = 0
            is_newborn = False
        else:
            old_x = _get_u16(memory, _POS_X)
            old_y = _get_u16(memory, _POS_Y)
            previous_age = self._get_age(memory)
            is_newborn = snapshot.age_days == 1 and previous_age > 1

            if is_newborn:
                self._rebase_inherited_memory(memory, old_x, old_y)
                old_x = old_y = 0
                memory[_PREVIOUS_ZONES] = self._zone_flags(snapshot)
            else:
                old_flags = memory[_PREVIOUS_ZONES]
                dx, dy = snapshot.velocity
                x, y = _wrap(old_x + dx), _wrap(old_y + dy)
                if len(memory) >= _ZONE_MEMORY_SIZE:
                    self._learn_crossings(
                        memory, old_x, old_y, dx, dy, old_flags, snapshot
                    )
                _put_u16(memory, _POS_X, x)
                _put_u16(memory, _POS_Y, y)
                memory[_PREVIOUS_ZONES] = self._zone_flags(snapshot)

            self._set_age(memory, snapshot.age_days)

        self._observe_current(memory, snapshot)
        x, y = _get_u16(memory, _POS_X), _get_u16(memory, _POS_Y)
        grid_size = self._grid_size(memory)
        if grid_size:
            self._mark_visited(memory, x, y, grid_size)

        self._refine_zone_centers(memory, x, y, snapshot)
        center = self._read_center(memory, target_zone)
        predator = self._read_center_info(memory, _PREDATOR)

        if snapshot.in_predator_zone and predator is None:
            return self._escape_without_predator_map(memory, snapshot)

        if center:
            target = self._safe_target(center, predator, (x, y))
            return self._plan(
                memory,
                (x, y),
                center,
                target,
                predator,
                snapshot.depth,
                in_target,
                snapshot.in_predator_zone,
            )

        if in_target and not snapshot.in_predator_zone:
            known_depths = [
                depth for depth in Depth if memory[_KNOWN_MASK] & (1 << depth.value)
            ]
            if known_depths:
                return min(
                    known_depths,
                    key=lambda depth: hypot(*self._velocity(memory, depth)),
                )

        return self._explore(memory, (x, y), snapshot)

    @staticmethod
    def _zone_flags(snapshot: JellyfishSnapshot) -> int:
        return (
            int(snapshot.in_feeding_zone)
            | (int(snapshot.in_breeding_zone) << 1)
            | (int(snapshot.in_predator_zone) << 2)
        )

    @staticmethod
    def _get_age(memory: bytearray) -> int:
        return int.from_bytes(memory[_LAST_AGE : _LAST_AGE + 4], "little")

    @staticmethod
    def _set_age(memory: bytearray, age: int) -> None:
        memory[_LAST_AGE : _LAST_AGE + 4] = min(age, 0xFFFFFFFF).to_bytes(4, "little")

    @staticmethod
    def _observe_current(memory: bytearray, snapshot: JellyfishSnapshot) -> None:
        depth_index = snapshot.depth.value
        offset = _VELOCITIES + depth_index * 2
        _put_i8(memory, offset, snapshot.velocity[0])
        _put_i8(memory, offset + 1, snapshot.velocity[1])
        memory[_KNOWN_MASK] |= 1 << depth_index

    @staticmethod
    def _velocity(memory: bytearray, depth: Depth) -> tuple[int, int]:
        offset = _VELOCITIES + depth.value * 2
        return _get_i8(memory, offset), _get_i8(memory, offset + 1)

    @staticmethod
    def _read_center(memory: bytearray, zone: int) -> tuple[float, float] | None:
        center_info = Player6._read_center_info(memory, zone)
        return center_info[0] if center_info else None

    @staticmethod
    def _read_center_info(
        memory: bytearray, zone: int
    ) -> tuple[tuple[float, float], int] | None:
        if len(memory) < _ZONE_MEMORY_SIZE:
            return None
        offset = _CENTERS + zone * 5
        quality = memory[offset + 4]
        if quality < 120:
            return None
        center = (float(_get_u16(memory, offset)), float(_get_u16(memory, offset + 2)))
        return center, quality

    @staticmethod
    def _safe_target(
        target: tuple[float, float],
        predator: tuple[tuple[float, float], int] | None,
        position: tuple[float, float],
    ) -> tuple[float, float]:
        if predator is None:
            return target

        predator_center, quality = predator
        safety_margin = max(45.0, 100.0 - 0.2 * quality)
        safe_radius = _ZONE_RADIUS + safety_margin
        dx = _delta(target[0], predator_center[0])
        dy = _delta(target[1], predator_center[1])
        separation = hypot(dx, dy)
        if separation >= safe_radius:
            return target
        if separation < 1:
            dx = _delta(position[0], predator_center[0])
            dy = _delta(position[1], predator_center[1])
            separation = hypot(dx, dy)
            if separation < 1:
                dx, dy, separation = 1.0, 0.0, 1.0

        unit_x, unit_y = dx / separation, dy / separation
        goal = (
            _wrap(predator_center[0] + unit_x * safe_radius),
            _wrap(predator_center[1] + unit_y * safe_radius),
        )
        if _distance(goal, target) > _ZONE_RADIUS:
            goal = (
                _wrap(target[0] + unit_x * _ZONE_RADIUS),
                _wrap(target[1] + unit_y * _ZONE_RADIUS),
            )
        return goal

    @staticmethod
    def _grid_size(memory: bytearray) -> int:
        if len(memory) >= _GRID_BASE + 20 * 20:
            return 20
        if len(memory) >= _GRID_BASE + 10 * 10:
            return 10
        return 0

    @staticmethod
    def _mark_visited(memory: bytearray, x: float, y: float, grid_size: int) -> None:
        cell_width = _WORLD / grid_size
        col = min(grid_size - 1, int(x / cell_width))
        row = min(grid_size - 1, int(y / cell_width))
        offset = _GRID_BASE + row * grid_size + col
        memory[offset] = min(memory[offset] + 1, 255)

    @staticmethod
    def _cell_count(memory: bytearray, x: float, y: float, grid_size: int) -> int:
        cell_width = _WORLD / grid_size
        col = min(grid_size - 1, int(x / cell_width))
        row = min(grid_size - 1, int(y / cell_width))
        return memory[_GRID_BASE + row * grid_size + col]

    @staticmethod
    def _rebase_inherited_memory(
        memory: bytearray, parent_x: int, parent_y: int
    ) -> None:
        if len(memory) >= _ZONE_MEMORY_SIZE:
            for zone in _ZONE_FLAGS:
                center_offset = _CENTERS + zone * 5
                if memory[center_offset + 4]:
                    _put_u16(
                        memory,
                        center_offset,
                        _get_u16(memory, center_offset) - parent_x,
                    )
                    _put_u16(
                        memory,
                        center_offset + 2,
                        _get_u16(memory, center_offset + 2) - parent_y,
                    )

                state = memory[_SAMPLE_STATES + zone]
                count = state & 0x07
                for sample in range(count):
                    offset = _SAMPLES + (zone * 4 + sample) * 4
                    _put_u16(memory, offset, _get_u16(memory, offset) - parent_x)
                    _put_u16(
                        memory,
                        offset + 2,
                        _get_u16(memory, offset + 2) - parent_y,
                    )
                if len(memory) >= _EXTRA_SAMPLE_MEMORY_SIZE:
                    extra_state = memory[_EXTRA_SAMPLE_STATES + zone]
                    extra_count = extra_state & 0x07
                    for sample in range(extra_count):
                        offset = _EXTRA_SAMPLES + (zone * 4 + sample) * 4
                        _put_u16(memory, offset, _get_u16(memory, offset) - parent_x)
                        _put_u16(
                            memory,
                            offset + 2,
                            _get_u16(memory, offset + 2) - parent_y,
                        )

        grid_size = Player6._grid_size(memory)
        if grid_size:
            start = _GRID_BASE
            end = start + grid_size * grid_size
            memory[start:end] = bytes(grid_size * grid_size)
        _put_u16(memory, _POS_X, 0)
        _put_u16(memory, _POS_Y, 0)

    def _learn_crossings(
        self,
        memory: bytearray,
        old_x: float,
        old_y: float,
        dx: float,
        dy: float,
        old_flags: int,
        snapshot: JellyfishSnapshot,
    ) -> None:
        new_flags = self._zone_flags(snapshot)
        for zone in _ZONE_FLAGS:
            was_inside = bool(old_flags & (1 << zone))
            is_inside = bool(new_flags & (1 << zone))
            if was_inside == is_inside:
                continue
            cross_x, cross_y = self._crossing_point(
                memory, zone, old_x, old_y, dx, dy, was_inside
            )
            self._add_boundary_sample(memory, zone, cross_x, cross_y)

    def _crossing_point(
        self,
        memory: bytearray,
        zone: int,
        old_x: float,
        old_y: float,
        dx: float,
        dy: float,
        was_inside: bool,
    ) -> tuple[float, float]:
        center_info = self._read_center_info(memory, zone)
        movement_squared = dx * dx + dy * dy
        if center_info is None or movement_squared == 0:
            return _wrap(old_x + dx / 2), _wrap(old_y + dy / 2)

        center, _ = center_info
        rel_x, rel_y = _delta(old_x, center[0]), _delta(old_y, center[1])
        linear = 2 * (rel_x * dx + rel_y * dy)
        constant = rel_x * rel_x + rel_y * rel_y - _ZONE_RADIUS**2
        discriminant = linear * linear - 4 * movement_squared * constant
        if discriminant < 0:
            return _wrap(old_x + dx / 2), _wrap(old_y + dy / 2)

        root = discriminant**0.5
        intersections = (
            (-linear - root) / (2 * movement_squared),
            (-linear + root) / (2 * movement_squared),
        )
        valid = [t for t in intersections if -0.02 <= t <= 1.02]
        if not valid:
            return _wrap(old_x + dx / 2), _wrap(old_y + dy / 2)

        fraction = min(valid) if not was_inside else max(valid)
        fraction = min(1.0, max(0.0, fraction))
        return _wrap(old_x + fraction * dx), _wrap(old_y + fraction * dy)

    def _add_boundary_sample(
        self, memory: bytearray, zone: int, x: float, y: float
    ) -> None:
        state_offset = _SAMPLE_STATES + zone
        state = memory[state_offset]
        count, cursor = state & 0x07, (state >> 3) & 0x03
        if count == 4 and len(memory) >= _EXTRA_SAMPLE_MEMORY_SIZE:
            extra_state_offset = _EXTRA_SAMPLE_STATES + zone
            extra_state = memory[extra_state_offset]
            extra_count, extra_cursor = (
                extra_state & 0x07,
                (extra_state >> 3) & 0x03,
            )
            sample_offset = _EXTRA_SAMPLES + (zone * 4 + extra_cursor) * 4
            _put_u16(memory, sample_offset, x)
            _put_u16(memory, sample_offset + 2, y)
            extra_count = min(4, extra_count + 1)
            extra_cursor = (extra_cursor + 1) % 4
            memory[extra_state_offset] = extra_count | (extra_cursor << 3)
            self._fit_zone_center(memory, zone)
            return

        sample_offset = _SAMPLES + (zone * 4 + cursor) * 4
        _put_u16(memory, sample_offset, x)
        _put_u16(memory, sample_offset + 2, y)
        count = min(4, count + 1)
        cursor = (cursor + 1) % 4
        memory[state_offset] = count | (cursor << 3)
        self._fit_zone_center(memory, zone)

    def _fit_zone_center(self, memory: bytearray, zone: int) -> None:
        state = memory[_SAMPLE_STATES + zone]
        count = state & 0x07
        points_count = count
        if len(memory) >= _EXTRA_SAMPLE_MEMORY_SIZE:
            points_count += memory[_EXTRA_SAMPLE_STATES + zone] & 0x07
        if points_count < 3:
            return
        points = []
        anchor_x = _get_u16(memory, _SAMPLES + zone * 16)
        anchor_y = _get_u16(memory, _SAMPLES + zone * 16 + 2)
        for sample in range(count):
            offset = _SAMPLES + (zone * 4 + sample) * 4
            px, py = _get_u16(memory, offset), _get_u16(memory, offset + 2)
            points.append(
                (
                    anchor_x + _delta(px, anchor_x),
                    anchor_y + _delta(py, anchor_y),
                )
            )
        if len(memory) >= _EXTRA_SAMPLE_MEMORY_SIZE:
            extra_count = memory[_EXTRA_SAMPLE_STATES + zone] & 0x07
            for sample in range(extra_count):
                offset = _EXTRA_SAMPLES + (zone * 4 + sample) * 4
                px, py = _get_u16(memory, offset), _get_u16(memory, offset + 2)
                points.append(
                    (
                        anchor_x + _delta(px, anchor_x),
                        anchor_y + _delta(py, anchor_y),
                    )
                )

        subsets = list(combinations(points, 3))
        if len(points) > 3:
            subsets.append(tuple(points))

        best_fit = None
        for subset in subsets:
            x0, y0 = subset[0]
            aa = ab = bb = ar = br = 0.0
            for px, py in subset[1:]:
                a, b = 2 * (px - x0), 2 * (py - y0)
                rhs = px * px + py * py - x0 * x0 - y0 * y0
                aa += a * a
                ab += a * b
                bb += b * b
                ar += a * rhs
                br += b * rhs
            determinant = aa * bb - ab * ab
            scale = aa * bb
            if determinant < 1_000_000 or scale == 0:
                continue
            condition = determinant / scale
            if condition < 0.035:
                continue

            center_x = (ar * bb - br * ab) / determinant
            center_y = (aa * br - ab * ar) / determinant
            errors = [
                abs(hypot(px - center_x, py - center_y) - _ZONE_RADIUS)
                for px, py in points
            ]
            inliers = [error for error in errors if error <= 50]
            inlier_points = [
                point for point, error in zip(points, errors) if error <= 50
            ]
            if len(inliers) < 3:
                continue
            spread = max(_distance(a, b) for a, b in combinations(inlier_points, 2))
            if spread < 100:
                continue
            rms_error = (sum(error * error for error in inliers) / len(inliers)) ** 0.5
            if rms_error > 45:
                continue
            fit_score = (len(inliers), -rms_error, condition)
            if best_fit is None or fit_score > best_fit[0]:
                best_fit = (
                    fit_score,
                    center_x,
                    center_y,
                    rms_error,
                    condition,
                )

        if best_fit is None:
            return

        support, _, condition = best_fit[0]
        center_x, center_y, rms_error = (
            best_fit[1],
            best_fit[2],
            best_fit[3],
        )
        center_offset = _CENTERS + zone * 5
        quality = min(
            255,
            round(180 + 30 * (support - 3) - 1.2 * rms_error + 20 * condition),
        )
        old_quality = memory[center_offset + 4]
        if old_quality:
            old_x = _get_u16(memory, center_offset)
            old_y = _get_u16(memory, center_offset + 2)
            shift = _distance((center_x, center_y), (old_x, old_y))
            if old_quality >= 170 and shift > 100 and support < 4:
                return
            blend = 0.25 if shift < 100 else 0.1
            center_x = old_x + blend * _delta(center_x, old_x)
            center_y = old_y + blend * _delta(center_y, old_y)
            quality = max(quality, old_quality - 10)
        _put_u16(memory, center_offset, center_x)
        _put_u16(memory, center_offset + 2, center_y)
        memory[center_offset + 4] = quality

    def _refine_zone_centers(
        self,
        memory: bytearray,
        x: float,
        y: float,
        snapshot: JellyfishSnapshot,
    ) -> None:
        if len(memory) < _ZONE_MEMORY_SIZE:
            return

        flags = self._zone_flags(snapshot)
        for zone in _ZONE_FLAGS:
            offset = _CENTERS + zone * 5
            quality = memory[offset + 4]
            if quality < 120:
                continue

            center_x = _get_u16(memory, offset)
            center_y = _get_u16(memory, offset + 2)
            dx, dy = _delta(x, center_x), _delta(y, center_y)
            distance = hypot(dx, dy)
            inside = bool(flags & (1 << zone))
            violation = distance - 205 if inside else 195 - distance
            if violation > 0 and distance > 0:
                correction = min(4.0, max(1.0, violation * 0.08))
                direction = 1 if inside else -1
                center_x += direction * correction * dx / distance
                center_y += direction * correction * dy / distance
                quality = max(0, quality - 1)
            else:
                quality = min(255, quality + 1)
            _put_u16(memory, offset, center_x)
            _put_u16(memory, offset + 2, center_y)
            memory[offset + 4] = quality

    def _escape_without_predator_map(
        self, memory: bytearray, snapshot: JellyfishSnapshot
    ) -> Depth:
        vx, vy = snapshot.velocity
        magnitude = hypot(vx, vy)
        if magnitude == 0:
            return snapshot.depth
        known_mask = memory[_KNOWN_MASK]
        known_depths = [depth for depth in Depth if known_mask & (1 << depth.value)]
        if not known_depths:
            return snapshot.depth
        return max(
            known_depths,
            key=lambda depth: (
                sum(a * b for a, b in zip(self._velocity(memory, depth), (vx, vy)))
                / magnitude
            ),
        )

    def _explore(
        self,
        memory: bytearray,
        position: tuple[float, float],
        snapshot: JellyfishSnapshot,
    ) -> Depth:
        known_mask = memory[_KNOWN_MASK]
        unknown = [depth for depth in Depth if not known_mask & (1 << depth.value)]
        if unknown:
            return self.rng.choice(unknown)

        grid_size = self._grid_size(memory)
        horizon = 12
        best_depth = snapshot.depth
        best_score = float("-inf")
        for depth in Depth:
            dx, dy = self._velocity(memory, depth)
            x, y = position
            visited: set[tuple[int, int]] = set()
            travel = 0.0
            for _ in range(horizon):
                x, y = _wrap(x + dx), _wrap(y + dy)
                travel += hypot(dx, dy)
                if grid_size:
                    cell_width = _WORLD / grid_size
                    visited.add(
                        (
                            min(grid_size - 1, int(x / cell_width)),
                            min(grid_size - 1, int(y / cell_width)),
                        )
                    )
            if grid_size:
                novelty = sum(
                    1
                    / (
                        1
                        + self._cell_count(
                            memory, col * cell_width, row * cell_width, grid_size
                        )
                    )
                    for col, row in visited
                )
                score = 20 * novelty + 0.03 * travel
            else:
                score = _distance(position, (x, y)) + 0.1 * travel
            if depth == snapshot.depth:
                score += 0.1
            if score > best_score:
                best_depth, best_score = depth, score
        return best_depth

    def _plan(
        self,
        memory: bytearray,
        position: tuple[float, float],
        target: tuple[float, float],
        goal: tuple[float, float],
        predator: tuple[tuple[float, float], int] | None,
        current_depth: Depth,
        already_in_target: bool,
        already_in_predator: bool,
    ) -> Depth:
        known_mask = memory[_KNOWN_MASK]
        known_velocities = {
            depth: self._velocity(memory, depth)
            for depth in Depth
            if known_mask & (1 << depth.value)
        }
        if not known_velocities:
            return current_depth

        beam: list[tuple[float, float, float, Depth | None, Depth]] = [
            (0.0, position[0], position[1], None, current_depth)
        ]
        beam_width, horizon = 12, 6

        for step in range(horizon):
            expanded = []
            for score, x, y, first_depth, last_depth in beam:
                for depth, (dx, dy) in known_velocities.items():
                    next_x, next_y = _wrap(x + dx), _wrap(y + dy)
                    target_cost = self._target_cost(
                        (next_x, next_y),
                        target,
                        goal,
                        already_in_target,
                    )
                    risk_cost = 0.0
                    if predator is not None:
                        predator_center, quality = predator
                        predator_distance = _distance((next_x, next_y), predator_center)
                        margin = max(45.0, 100.0 - 0.2 * quality)
                        safe_radius = _ZONE_RADIUS + margin
                        if predator_distance <= _ZONE_RADIUS:
                            risk_cost = 1200.0 + 2.0 * (
                                _ZONE_RADIUS - predator_distance
                            )
                        elif predator_distance < safe_radius:
                            risk_fraction = (safe_radius - predator_distance) / margin
                            risk_cost = 600.0 * risk_fraction**2
                        if (
                            step == 0
                            and already_in_predator
                            and predator_distance <= _ZONE_RADIUS
                        ):
                            risk_cost += 500.0
                    switch_cost = 3.0 if depth != last_depth else 0.0
                    expanded.append(
                        (
                            score + target_cost + risk_cost + switch_cost,
                            next_x,
                            next_y,
                            depth if first_depth is None else first_depth,
                            depth,
                        )
                    )
            beam = sorted(expanded, key=lambda node: node[0])[:beam_width]

        best = min(
            beam,
            key=lambda node: (
                node[0]
                + 1.5
                * self._target_cost(
                    (node[1], node[2]),
                    target,
                    goal,
                    already_in_target,
                )
            ),
        )
        return best[3] if best[3] is not None else current_depth

    @staticmethod
    def _target_cost(
        position: tuple[float, float],
        center: tuple[float, float],
        goal: tuple[float, float],
        already_in_target: bool,
    ) -> float:
        center_distance = _distance(position, center)
        preferred_radius_cost = 0.01 * (center_distance - 80.0) ** 2
        outside_cost = max(0.0, center_distance - 190.0) * (
            4.0 if already_in_target else 3.0
        )
        return preferred_radius_cost + 0.35 * _distance(position, goal) + outside_cost
