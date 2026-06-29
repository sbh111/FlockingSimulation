"""
Author: Saad Bhatti
Desc:
A vectorized, Structure-of-Arrays (SoA) implementation of the flocking algorithm.

Instead of one Python object per boid (see boid.py / flock.py), the entire flock
is held in a handful of contiguous arrays (positions, velocities, per-boid radius,
etc.). One simulation step is expressed as array math over an N x N distance
matrix, so the whole flock is updated at once rather than boid-by-boid.

The array module is parameterized as ``xp``. With ``xp = numpy`` this runs on the
CPU; in Phase 2 the same code runs on the GPU by passing ``xp = cupy``. Nothing in
the step logic below is NumPy-specific.

Two intentional differences from the object-based version:

  * Update is *simultaneous* (Jacobi style): every boid computes its acceleration
    from the previous frame's state for all boids. The object version mutates each
    boid in place mid-loop (Gauss-Seidel), so later boids see earlier boids'
    already-updated positions. Jacobi is order-independent and the natural fit for
    vectorized/GPU execution; the emergent flocking behavior is equivalent.

  * Neighbor search is a brute-force O(N^2) distance matrix rather than a quad-tree.
    The quad-tree is a serial, pointer-chasing structure that does not vectorize; on
    arrays the dense matrix is faster up to fairly large N (and is the basis for the
    GPU path). The per-boid heterogeneous neighbor radius is just a per-row
    threshold, so it carries over unchanged.

The numeric constants (rule weights, radius/speed distributions, wrap behavior)
mirror boid.py and flock.py so behavior matches the original.
"""

import numpy as np


# Separation weight is inversely proportional to the flock size: the per-frame
# weight is SEPARATION_K / N. Calibrated to equal the previous fixed 1.1 at the
# default 350-boid flock (1.1 * 350 = 385). Matches flock.py's SEPARATION_K so
# the two backends behave identically.
SEPARATION_K = 385.0


class FlockArray:
    def __init__(self, n, width, height, seed=None, xp=np, dtype=None,
                 use_grid=False):
        self.xp = xp
        self.w = float(width)
        self.h = float(height)
        self.dtype = dtype if dtype is not None else xp.float32
        # Neighbour search: False -> brute-force O(N^2) distance matrix;
        # True -> uniform spatial-hash grid (memory-light, scales past the
        # dense matrix's VRAM ceiling). See _accumulate_grid.
        self.use_grid = use_grid

        self._init_state(n, seed)

    @classmethod
    def from_boids(cls, boids, width, height, xp=np, dtype=None, use_grid=False):
        """Build a flock from existing Boid objects (flock.py), copying every
        per-boid attribute so the array backend continues the *same* flock rather
        than a fresh random one. Used when toggling backends at runtime."""
        obj = cls.__new__(cls)
        obj.xp = xp
        obj.w = float(width)
        obj.h = float(height)
        obj.dtype = dtype if dtype is not None else xp.float32
        obj.use_grid = use_grid
        dt = obj.dtype
        obj.n = len(boids)
        obj._rng = np.random.default_rng()  # host RNG (see _init_state)

        if obj.n == 0:
            obj.pos = xp.zeros((0, 2), dtype=dt)
            obj.vel = xp.zeros((0, 2), dtype=dt)
            obj.size = xp.zeros((0,), dtype=xp.int32)
            obj.radius = xp.zeros((0,), dtype=dt)
            obj.radius2 = xp.zeros((0,), dtype=dt)
            obj.max_speed = xp.zeros((0,), dtype=dt)
            obj.color = xp.zeros((0, 3), dtype=xp.int32)
            return obj

        obj.pos = xp.asarray([[b.pos.x, b.pos.y] for b in boids], dtype=dt)
        obj.vel = xp.asarray([[b.velocity.x, b.velocity.y] for b in boids], dtype=dt)
        obj.size = xp.asarray([b.s for b in boids], dtype=xp.int32)
        radius = xp.asarray([b.neighborRadius for b in boids], dtype=dt)
        obj.radius = radius
        obj.radius2 = (radius * radius).astype(dt)
        obj.max_speed = xp.asarray([b.maxSpeed for b in boids], dtype=dt)
        obj.color = xp.asarray([list(b.color) for b in boids], dtype=xp.int32)
        return obj

    def write_to_flock(self, flock):
        """Replace `flock.flock` with Boid objects rebuilt from this array's
        state, so switching back to the object backend continues the same flock.
        Handles boids added/removed while in array mode."""
        import pygame.math as m
        from boid import Boid

        P = self.positions_host()
        V = self.vel if self.xp is np else self.xp.asnumpy(self.vel)
        size = self.size if self.xp is np else self.xp.asnumpy(self.size)
        radius = self.radius if self.xp is np else self.xp.asnumpy(self.radius)
        max_speed = self.max_speed if self.xp is np else self.xp.asnumpy(self.max_speed)
        color = self.color if self.xp is np else self.xp.asnumpy(self.color)

        boids = []
        for i in range(self.n):
            b = Boid(m.Vector2(float(P[i, 0]), float(P[i, 1])),
                     m.Vector2(float(V[i, 0]), float(V[i, 1])))
            b.s = int(size[i])
            b.maxSpeed = float(max_speed[i])
            b.neighborRadius = float(radius[i])
            b.color = (int(color[i, 0]), int(color[i, 1]), int(color[i, 2]))
            # keep the boid's query Circle consistent with the copied attributes
            b.circle.x = b.x
            b.circle.y = b.y
            b.circle.radius = b.neighborRadius
            boids.append(b)
        flock.flock = boids

    # ------------------------------------------------------------------ setup
    def _init_state(self, n, seed):
        xp = self.xp
        dt = self.dtype
        # The one-time random init is done on the host with NumPy and then moved
        # to the device. This sidesteps API differences between numpy.random and
        # cupy.random (e.g. CuPy's Generator lacks .normal) and makes the initial
        # flock identical across the CPU and GPU backends.
        rng = np.random.default_rng(seed)

        self.n = int(n)

        # Positions uniformly across the screen; velocities in [-10, 10] per axis,
        # matching Flock.insertBoid / Boid.__init__.
        px = rng.uniform(0, self.w, self.n)
        py = rng.uniform(0, self.h, self.n)
        self.pos = xp.asarray(np.stack([px, py], axis=1), dtype=dt)

        vx = rng.uniform(-10, 10, self.n)
        vy = rng.uniform(-10, 10, self.n)
        self.vel = xp.asarray(np.stack([vx, vy], axis=1), dtype=dt)

        # Per-boid line thickness s in {1,2,3,4} (int of uniform[1,5)).
        s = rng.uniform(1, 5, self.n).astype(np.int32)
        self.size = xp.asarray(s)

        # neighborRadius = max(s*10, int(gauss(30, 20))), matching Boid.__init__.
        r = rng.normal(30, 20, self.n).astype(np.int32)
        radius = np.maximum(s * 10, r).astype(np.float32)
        self.radius = xp.asarray(radius, dtype=dt)
        self.radius2 = (self.radius * self.radius).astype(dt)

        # maxSpeed = abs(gauss(20, 5)).
        self.max_speed = xp.asarray(np.abs(rng.normal(20, 5, self.n)), dtype=dt)

        self.color = xp.asarray(rng.integers(0, 256, size=(self.n, 3)).astype(np.int32))

        # A reusable host RNG for the per-frame "wander" jitter and add_boid.
        self._rng = rng

    def add_boid(self, x, y):
        """Append a single boid at (x, y) with the same random attribute draws
        the constructor uses. Kept simple; rebuilds the derived arrays."""
        xp = self.xp
        dt = self.dtype
        rng = self._rng

        vx = rng.uniform(-10, 10)
        vy = rng.uniform(-10, 10)
        s = int(rng.uniform(1, 5))
        r = int(rng.normal(30, 20))
        radius = max(s * 10, r)

        self.pos = xp.concatenate([self.pos, xp.asarray([[x, y]], dtype=dt)], axis=0)
        self.vel = xp.concatenate([self.vel, xp.asarray([[vx, vy]], dtype=dt)], axis=0)
        self.size = xp.concatenate([self.size, xp.asarray([s], dtype=xp.int32)])
        self.radius = xp.concatenate([self.radius, xp.asarray([radius], dtype=dt)])
        self.radius2 = (self.radius * self.radius).astype(dt)
        self.max_speed = xp.concatenate(
            [self.max_speed, xp.asarray([abs(rng.normal(20, 5))], dtype=dt)])
        rgb = xp.asarray(rng.integers(0, 256, size=(1, 3)).astype(np.int32))
        self.color = xp.concatenate([self.color, rgb], axis=0)
        self.n += 1

    def remove_boid(self):
        """Drop the oldest boid (front of the arrays), matching Flock.removeBoid's
        pop(0) semantics."""
        if self.n == 0:
            return
        self.pos = self.pos[1:]
        self.vel = self.vel[1:]
        self.size = self.size[1:]
        self.radius = self.radius[1:]
        self.radius2 = self.radius2[1:]
        self.max_speed = self.max_speed[1:]
        self.color = self.color[1:]
        self.n -= 1

    # ------------------------------------------------------------------- step
    def step(self, mouse_pos=None, useCohesion=True, useSeparation=True,
             useAlignment=True):
        """Advance the whole flock by one frame. Pure array math, no rendering."""
        if self.n == 0:
            return
        xp = self.xp
        dt = self.dtype
        P = self.pos
        V = self.vel
        n = self.n

        # Neighbour accumulation: per boid, the count and the sums needed by the
        # three rules. Both backends below return the same quantities, so the rule
        # combination is identical regardless of how neighbours were found.
        if self.use_grid:
            count, sum_pos, sum_vel, sep = self._accumulate_grid()
        else:
            count, sum_pos, sum_vel, sep = self._accumulate_bruteforce()

        has_n = count > 0
        safe_count = xp.where(has_n, count, 1.0)              # avoid div-by-zero

        acc = xp.zeros((n, 2), dtype=dt)

        if useCohesion:
            # 0.004 * (mean(neighbor positions) - own position).
            center = sum_pos / safe_count[:, None]
            acc = acc + 0.004 * (center - P)

        if useAlignment:
            # 0.5 * (mean(neighbor velocities) - own velocity).
            mean_vel = sum_vel / safe_count[:, None]
            acc = acc + 0.5 * (mean_vel - V)

        if useSeparation:
            # (SEPARATION_K / N) * sum_j (P[i] - P[j]) / dist2; weight scales
            # inversely with the total flock size N.
            sep_factor = SEPARATION_K / n
            acc = acc + sep_factor * sep

        # Boids with no neighbors get zero rule-acceleration (original returns 0).
        acc = acc * has_n[:, None]

        if mouse_pos is not None:
            acc = acc + self._avoid_mouse(mouse_pos)

        # Wander: boids whose total acceleration is exactly zero get a small random
        # nudge (matches Flock.draw's `if acc == Vector2(0, 0)` branch).
        zero = (acc[:, 0] == 0) & (acc[:, 1] == 0)
        if bool(zero.any()):
            k = int(zero.sum())
            jitter = xp.asarray(self._rng.uniform(-2, 2, size=(k, 2)), dtype=dt)
            acc[zero] = acc[zero] + 0.3 * jitter

        # Integrate: v += a; clamp speed; p += v; wrap.
        V = V + acc
        speed = xp.sqrt((V * V).sum(axis=1))
        safe_speed = xp.where(speed > 0, speed, 1.0)
        over = (speed > self.max_speed) & (speed > 0)
        scale = xp.where(over, self.max_speed / safe_speed, 1.0).astype(dt)
        V = V * scale[:, None]

        P = P + V
        # Wrap: only when strictly past the edge (matches Boid.wrapAround exactly).
        px = P[:, 0]
        py = P[:, 1]
        px = xp.where(px < 0, self.w, px)
        px = xp.where(px > self.w, 0.0, px)
        py = xp.where(py < 0, self.h, py)
        py = xp.where(py > self.h, 0.0, py)
        P = xp.stack([px, py], axis=1).astype(dt)

        self.pos = P
        self.vel = V

    # ------------------------------------------------------- neighbour search
    def _accumulate_bruteforce(self):
        """Neighbour sums via the dense O(N^2) pairwise distance matrix.
        Returns (count, sum_pos, sum_vel, sep) where sum_pos/sum_vel are (N,2)
        sums over each boid's neighbours and sep is the (N,2) separation vector."""
        xp = self.xp
        dt = self.dtype
        P = self.pos
        V = self.vel
        n = self.n

        dx = P[:, 0][:, None] - P[:, 0][None, :]            # (N, N)
        dy = P[:, 1][:, None] - P[:, 1][None, :]
        dist2 = dx * dx + dy * dy

        # neighbor[i, j]: j within boid i's (asymmetric) radius and j != i.
        neighbor = dist2 <= self.radius2[:, None]
        idx = xp.arange(n)
        neighbor[idx, idx] = False

        neighbor_f = neighbor.astype(dt)
        count = neighbor_f.sum(axis=1)
        sum_pos = neighbor_f @ P
        sum_vel = neighbor_f @ V

        safe_d2 = xp.where(dist2 > 0, dist2, 1.0)
        inv = xp.where(neighbor & (dist2 > 0), 1.0 / safe_d2, 0.0).astype(dt)
        sep = xp.stack([(inv * dx).sum(axis=1), (inv * dy).sum(axis=1)], axis=1)
        return count, sum_pos, sum_vel, sep

    def _accumulate_grid(self):
        """Neighbour sums via a uniform spatial-hash grid. Equivalent result to
        _accumulate_bruteforce but never materialises an N x N array, so it scales
        to far larger flocks within a small GPU's VRAM.

        Cells are sized at the largest neighbour radius (R_max), so every boid's
        neighbours (radius <= R_max) lie within its own cell or the 8 around it
        (a 3x3 block). Boids are bucketed into a dense (n_cells, max_occupancy)
        table of indices via a counting sort; the 3x3 candidates are then gathered
        and filtered by each boid's actual radius. Work scales with cell occupancy
        rather than N^2 (worst case degrades only if boids pile into one cell)."""
        xp = self.xp
        dt = self.dtype
        P = self.pos
        V = self.vel
        n = self.n
        px = P[:, 0]
        py = P[:, 1]
        vx = V[:, 0]
        vy = V[:, 1]

        cell_size = float(self.radius.max())
        if cell_size <= 0:
            cell_size = 1.0
        gw = max(1, int(np.ceil(self.w / cell_size)))
        gh = max(1, int(np.ceil(self.h / cell_size)))
        ncells = gw * gh

        cx = xp.clip((px / cell_size).astype(xp.int64), 0, gw - 1)
        cy = xp.clip((py / cell_size).astype(xp.int64), 0, gh - 1)
        cell = cy * gw + cx                                  # (N,)

        # Counting sort into a (ncells, maxslots) table of boid indices (-1 pad).
        order = xp.argsort(cell)
        sorted_cell = cell[order]
        counts = xp.bincount(cell, minlength=ncells)
        maxslots = int(counts.max())
        cell_start = xp.zeros(ncells, dtype=xp.int64)
        cell_start[1:] = xp.cumsum(counts)[:-1]
        slot = xp.arange(n) - cell_start[sorted_cell]
        table = xp.full((ncells, maxslots), -1, dtype=xp.int64)
        table[sorted_cell, slot] = order

        count = xp.zeros(n, dtype=dt)
        sum_px = xp.zeros(n, dtype=dt)
        sum_py = xp.zeros(n, dtype=dt)
        sum_vx = xp.zeros(n, dtype=dt)
        sum_vy = xp.zeros(n, dtype=dt)
        sep_x = xp.zeros(n, dtype=dt)
        sep_y = xp.zeros(n, dtype=dt)
        r2c = self.radius2[:, None]
        my_idx = xp.arange(n)[:, None]

        # Visit the 3x3 block of cells around each boid. Each offset maps to a
        # distinct neighbour cell (out-of-range offsets are masked, not clamped,
        # so edge cells are never double-counted). The slot dimension is fully
        # vectorised: for each offset we gather every boid's whole candidate cell
        # at once as an (N, maxslots) array and reduce, so the per-frame kernel
        # count stays small (~9 offsets) instead of 9 * maxslots.
        for ox in (-1, 0, 1):
            nbx = cx + ox
            in_x = (nbx >= 0) & (nbx < gw)
            for oy in (-1, 0, 1):
                nby = cy + oy
                in_range = in_x & (nby >= 0) & (nby < gh)    # (N,)
                ncell = xp.where(in_range, nby * gw + nbx, 0)
                J = table[ncell]                             # (N, maxslots)
                valid = in_range[:, None] & (J >= 0) & (J != my_idx)
                JJ = xp.where(valid, J, 0)                    # safe gather indices
                ddx = px[:, None] - px[JJ]                    # (N, maxslots)
                ddy = py[:, None] - py[JJ]
                d2 = ddx * ddx + ddy * ddy
                within = valid & (d2 <= r2c)
                w = within.astype(dt)
                count = count + w.sum(axis=1)
                sum_px = sum_px + (w * px[JJ]).sum(axis=1)
                sum_py = sum_py + (w * py[JJ]).sum(axis=1)
                sum_vx = sum_vx + (w * vx[JJ]).sum(axis=1)
                sum_vy = sum_vy + (w * vy[JJ]).sum(axis=1)
                inv = xp.where(within & (d2 > 0),
                               1.0 / xp.where(d2 > 0, d2, 1.0), 0.0).astype(dt)
                sep_x = sep_x + (inv * ddx).sum(axis=1)
                sep_y = sep_y + (inv * ddy).sum(axis=1)

        sum_pos = xp.stack([sum_px, sum_py], axis=1)
        sum_vel = xp.stack([sum_vx, sum_vy], axis=1)
        sep = xp.stack([sep_x, sep_y], axis=1)
        return count, sum_pos, sum_vel, sep

    def _avoid_mouse(self, mouse_pos):
        """Vectorized port of Flock.avoidMouse: boids whose neighbor-circle contains
        the mouse flee from it, with the same dot-product sign correction."""
        xp = self.xp
        dt = self.dtype
        P = self.pos
        V = self.vel
        mx, my = float(mouse_pos[0]), float(mouse_pos[1])

        to_mouse_x = P[:, 0] - mx
        to_mouse_y = P[:, 1] - my
        dist2 = to_mouse_x * to_mouse_x + to_mouse_y * to_mouse_y
        active = dist2 <= self.radius2                       # mouse inside circle

        # acc = (pos + vel) - mouse
        ax = (P[:, 0] + V[:, 0]) - mx
        ay = (P[:, 1] + V[:, 1]) - my

        # If acc . vel < 0, zero one component based on sign agreement with vel.x.
        dot = ax * V[:, 0] + ay * V[:, 1]
        neg = dot < 0
        same_sign = xp.sign(V[:, 0]) == xp.sign(ax)
        # not same sign -> acc.x = 0 ; else acc.y = 0
        ax = xp.where(neg & ~same_sign, 0.0, ax)
        ay = xp.where(neg & same_sign, 0.0, ay)

        dist = xp.sqrt(dist2)
        safe_dist = xp.where(dist > 0, dist, 1.0)
        factor = xp.where(dist > 0, 1.0 / safe_dist, 100.0)
        ax = ax * factor
        ay = ay * factor

        out = xp.stack([ax, ay], axis=1).astype(dt)
        return out * active[:, None]

    # --------------------------------------------------------------- rendering
    def positions_host(self):
        """Return positions as a host (NumPy) array, copying off-device if needed."""
        P = self.pos
        if self.xp is not np:
            P = self.xp.asnumpy(P)
        return P

    def draw(self):
        """Per-boid pygame rendering. This stays a Python loop on purpose; it is the
        Amdahl ceiling discussed in the plan and is not what we're optimizing here."""
        import pygame
        screen = pygame.display.get_surface()

        P = self.positions_host()
        V = self.vel
        if self.xp is not np:
            V = self.xp.asnumpy(V)
        colors = self.color if self.xp is np else self.xp.asnumpy(self.color)
        sizes = self.size if self.xp is np else self.xp.asnumpy(self.size)

        speeds = np.sqrt((V * V).sum(axis=1))
        for i in range(self.n):
            sp = speeds[i]
            if sp == 0:
                continue
            nx = V[i, 0] / sp
            ny = V[i, 1] / sp
            start = (int(P[i, 0]), int(P[i, 1]))
            end = (float(P[i, 0] + 15 * nx), float(P[i, 1] + 15 * ny))
            pygame.draw.line(screen, tuple(int(c) for c in colors[i]),
                             start, end, int(sizes[i]))
