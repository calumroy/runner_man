import math
import sys
import random
import pygame


WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540
FPS = 60

# World physics
GRAVITY_PIXELS_PER_S2 = 1400.0
AIR_RESISTANCE = 0.12
ANGULAR_DAMPING = 2.2
PENETRATION_SLOP = 0.1
PENETRATION_CORRECT_PERCENT = 0.85
COLLISION_SPIN_COEFF = 0.006
MAX_ANGULAR_SPEED_RAD = 3.0
PLATFORM_GROUND_ANGLE_LIMIT_DEG = 18.0
SLIDE_ACCEL_FACTOR = 0.7
PLATFORM_COLLISION_SPIN_BOUNCE = 0.35
PLATFORM_COLLISION_RESTITUTION = 0.05
PLATFORM_COLLISION_FRICTION = 0.2
SOLVER_ITERATIONS = 12
DEEP_PENETRATION_THRESHOLD = 4.0
SEPARATION_BIAS = 0.2
PLATFORM_FRICTION_COEFF = 0.6
STABILIZE_AFTER_EXPLOSION_PASSES = 3

# Player tuning
PLAYER_WIDTH = 36
PLAYER_HEIGHT = 52
PLAYER_MASS_KG = 70.0
PLAYER_MOVE_ACCEL = 3200.0
PLAYER_MAX_RUN_SPEED = 480.0
PLAYER_JUMP_SPEED = 680.0
GROUND_FRICTION = 0.3

# Jump feel helpers
COYOTE_TIME_S = 0.12
JUMP_BUFFER_S = 0.12
JUMP_CUTOFF_SPEED = 240.0

# Missile tuning
MISSILE_WIDTH = 24
MISSILE_HEIGHT = 12
MISSILE_MASS_KG = 20.0
MISSILE_THRUST = 2000.0
MISSILE_MAX_SPEED = 220.0
MISSILE_TURN_RATE_DEG_PER_S = 60.0
MISSILE_SPAWN_INTERVAL_START_S = 4.0
MISSILE_SPAWN_INTERVAL_MIN_S = 0.7
MISSILE_SPAWN_INTERVAL_DECAY_PER_MIN = 1.8

# Explosions
EXPLOSION_MIN_DELAY_S = 2.2
EXPLOSION_MAX_DELAY_S = 4.8
EXPLOSION_RADIUS = 240.0
EXPLOSION_IMPULSE = 300000.0
EXPLOSION_LIFE_S = 0.35
EXPLOSION_TORQUE_FACTOR = 3.0
EXPLOSION_TANGENTIAL_FRACTION = 0.35
SHIELD_PUSH_IMPULSE = 60000.0
SHIELD_TORQUE_FACTOR = 1.2


Color = pygame.Color


class GameObject:
    def __init__(
        self,
        rect: pygame.Rect,
        color: Color,
        mass_kg: float,
        is_anchored: bool = False,
        restitution: float = 0.0,
    ) -> None:
        self.rect = rect
        self.color = color
        self.mass_kg = mass_kg
        self.is_anchored = is_anchored
        self.restitution = max(0.0, min(1.0, restitution))
        self.position = pygame.Vector2(rect.x, rect.y)
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)
        self.on_ground = False
        # Rotation (radians) for objects that choose to render with rotation
        self.angle_rad = 0.0
        self.angular_velocity = 0.0

    def apply_force(self, force: pygame.Vector2) -> None:
        if self.is_anchored:
            return
        # a = F / m
        self.acceleration += (force / max(self.mass_kg, 1e-6))

    def apply_gravity(self) -> None:
        if not self.is_anchored:
            self.apply_force(pygame.Vector2(0.0, GRAVITY_PIXELS_PER_S2 * self.mass_kg))

    def integrate(self, dt: float) -> None:
        if self.is_anchored:
            # Keep anchored objects fixed in place, but still allow them to have mass for interactions
            self.acceleration.update(0.0, 0.0)
            self.velocity.update(0.0, 0.0)
            return
        # Apply air drag
        drag = -self.velocity * AIR_RESISTANCE
        self.apply_force(drag * self.mass_kg)
        # Integrate using semi-implicit Euler
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        self.rect.topleft = (int(self.position.x), int(self.position.y))
        # Angular integration with damping
        if self.angular_velocity != 0.0:
            # Clamp angular speed
            if self.angular_velocity > MAX_ANGULAR_SPEED_RAD:
                self.angular_velocity = MAX_ANGULAR_SPEED_RAD
            elif self.angular_velocity < -MAX_ANGULAR_SPEED_RAD:
                self.angular_velocity = -MAX_ANGULAR_SPEED_RAD
            self.angle_rad += self.angular_velocity * dt
            # simple angular damping
            self.angular_velocity *= max(0.0, (1.0 - ANGULAR_DAMPING * dt))
        # Reset forces
        self.acceleration.update(0.0, 0.0)

    def apply_impulse(self, impulse: pygame.Vector2) -> None:
        if self.is_anchored:
            return
        # Impulse changes velocity instantaneously: dv = J / m
        self.velocity += impulse / max(self.mass_kg, 1e-6)

    def moment_of_inertia(self) -> float:
        # Approximate as rectangle about center
        w = float(self.rect.width)
        h = float(self.rect.height)
        return (self.mass_kg * (w * w + h * h)) / 12.0 if self.mass_kg > 0 else 0.0

    def apply_angular_impulse(self, angular_impulse: float) -> None:
        if self.is_anchored:
            return
        I = self.moment_of_inertia()
        if I > 1e-6:
            self.angular_velocity += angular_impulse / I


class Platform(GameObject):
    pass


class MovingPlatform(Platform):
    def __init__(self, rect: pygame.Rect, color: Color, mass_kg: float, velocity: pygame.Vector2) -> None:
        super().__init__(rect, color, mass_kg, is_anchored=False, restitution=0.2)
        self.velocity = velocity
        self.max_speed = 360.0
        self.angle_rad = 0.0
        self.angular_velocity = 0.0

    def update_motion_and_bounce(self, dt: float) -> None:
        # Clamp speed to avoid runaway
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        # Integrate forces and velocity have already been applied via integrate
        # Bounce off screen bounds (horizontal walls and ceiling)
        bounced = False
        if self.rect.left < 0:
            self.position.x = 0
            self.rect.left = 0
            self.velocity.x = -self.velocity.x * 0.9
            self.angular_velocity = -self.angular_velocity * (0.5 + 0.5 * self.restitution)
            bounced = True
        elif self.rect.right > WINDOW_WIDTH:
            self.position.x = WINDOW_WIDTH - self.rect.width
            self.rect.right = WINDOW_WIDTH
            self.velocity.x = -self.velocity.x * 0.9
            self.angular_velocity = -self.angular_velocity * (0.5 + 0.5 * self.restitution)
            bounced = True
        if self.rect.top < 0:
            self.position.y = 0
            self.rect.top = 0
            self.velocity.y = -self.velocity.y * 0.9
            self.angular_velocity = -self.angular_velocity * (0.5 + 0.5 * self.restitution)
            bounced = True
        if self.rect.bottom > WINDOW_HEIGHT:
            self.position.y = WINDOW_HEIGHT - self.rect.height
            self.rect.bottom = WINDOW_HEIGHT
            self.velocity.y = -self.velocity.y * 0.9
            self.angular_velocity = -self.angular_velocity * (0.5 + 0.5 * self.restitution)
            bounced = True
        # If we bounced, slightly reduce speed
        if bounced:
            self.velocity *= 0.9


class Player(GameObject):
    def __init__(self, rect: pygame.Rect) -> None:
        super().__init__(rect, Color(50, 200, 255), PLAYER_MASS_KG, is_anchored=False, restitution=0.0)
        self.lives = 3
        self.invuln_timer_s = 0.0
        self.facing = pygame.Vector2(1, 0)
        self.shield_cooldown_s = 0.0

    def move_left(self, dt: float) -> None:
        desired = -PLAYER_MAX_RUN_SPEED
        if self.velocity.x > desired:
            self.apply_force(pygame.Vector2(-PLAYER_MOVE_ACCEL * self.mass_kg, 0.0))
        self.facing.update(-1, 0)

    def move_right(self, dt: float) -> None:
        desired = PLAYER_MAX_RUN_SPEED
        if self.velocity.x < desired:
            self.apply_force(pygame.Vector2(PLAYER_MOVE_ACCEL * self.mass_kg, 0.0))
        self.facing.update(1, 0)

    def jump(self) -> None:
        if self.on_ground:
            self.velocity.y = -PLAYER_JUMP_SPEED
            self.on_ground = False

    def take_hit(self) -> None:
        if self.invuln_timer_s > 0.0:
            return
        self.lives = max(0, self.lives - 1)
        self.invuln_timer_s = 1.0
        # Small knockback upwards to telegraph hit
        self.velocity.y = min(self.velocity.y, -PLAYER_JUMP_SPEED * 0.6)


class Missile(GameObject):
    def __init__(self, rect: pygame.Rect, explode_at_s: float) -> None:
        super().__init__(rect, Color(255, 90, 70), MISSILE_MASS_KG, is_anchored=False, restitution=0.2)
        self.heading_rad = 0.0
        self.explode_at_s = explode_at_s

    def update_heading_towards(self, target_pos: pygame.Vector2, dt: float) -> None:
        to_target = (target_pos - pygame.Vector2(self.rect.center)).normalize() if target_pos != pygame.Vector2(self.rect.center) else pygame.Vector2(1, 0)
        desired_angle = math.atan2(to_target.y, to_target.x)
        # Shortest angular difference
        delta = (desired_angle - self.heading_rad + math.pi) % (2 * math.pi) - math.pi
        max_turn = math.radians(MISSILE_TURN_RATE_DEG_PER_S) * dt
        if delta > max_turn:
            delta = max_turn
        elif delta < -max_turn:
            delta = -max_turn
        self.heading_rad += delta

    def apply_thrust(self) -> None:
        forward = pygame.Vector2(math.cos(self.heading_rad), math.sin(self.heading_rad))
        self.apply_force(forward * MISSILE_THRUST * self.mass_kg)

    def clamp_speed(self) -> None:
        speed = self.velocity.length()
        if speed > MISSILE_MAX_SPEED:
            self.velocity.scale_to_length(MISSILE_MAX_SPEED)


def aabb_resolve_dynamic_static(dynamic: GameObject, static: GameObject) -> None:
    if dynamic.rect.colliderect(static.rect):
        # Compute overlap
        dx1 = static.rect.right - dynamic.rect.left
        dx2 = dynamic.rect.right - static.rect.left
        dy1 = static.rect.bottom - dynamic.rect.top
        dy2 = dynamic.rect.bottom - static.rect.top
        overlap_x = dx1 if abs(dx1) < abs(dx2) else -dx2
        overlap_y = dy1 if abs(dy1) < abs(dy2) else -dy2

        # Resolve along minimal axis
        if abs(overlap_x) < abs(overlap_y):
            dynamic.position.x += overlap_x
            dynamic.rect.x = int(dynamic.position.x)
            dynamic.velocity.x = -dynamic.velocity.x * dynamic.restitution
        else:
            dynamic.position.y += overlap_y
            dynamic.rect.y = int(dynamic.position.y)
            hitting_from_top = overlap_y < 0
            if hitting_from_top:
                angle_limit_rad = math.radians(PLATFORM_GROUND_ANGLE_LIMIT_DEG)
                if not isinstance(static, MovingPlatform) or abs(static.angle_rad) <= angle_limit_rad:
                    dynamic.on_ground = True
            # Bounce with restitution
            dynamic.velocity.y = -dynamic.velocity.y * dynamic.restitution


def aabb_resolve_dynamic_dynamic(a: GameObject, b: GameObject) -> None:
    if not a.rect.colliderect(b.rect):
        return
    # Compute overlap amounts in both axes
    dx1 = b.rect.right - a.rect.left
    dx2 = a.rect.right - b.rect.left
    dy1 = b.rect.bottom - a.rect.top
    dy2 = a.rect.bottom - b.rect.top
    overlap_x = dx1 if abs(dx1) < abs(dx2) else -dx2
    overlap_y = dy1 if abs(dy1) < abs(dy2) else -dy2

    # Choose minimal axis to resolve
    sep_axis_is_x = abs(overlap_x) < abs(overlap_y)
    total_mass = max(a.mass_kg + b.mass_kg, 1e-6)

    if sep_axis_is_x:
        # Separate along X using Baumgarte positional correction
        penetration = abs(overlap_x)
        correction = (penetration - PENETRATION_SLOP)
        if correction < 0:
            correction = 0
        correction *= PENETRATION_CORRECT_PERCENT
        # Additional strong bias for deep penetrations
        if penetration > DEEP_PENETRATION_THRESHOLD:
            correction += (penetration - DEEP_PENETRATION_THRESHOLD) * (0.5 + SEPARATION_BIAS)
        sign = 1 if overlap_x > 0 else -1
        move_a = -sign * correction * (b.mass_kg / total_mass)
        move_b = sign * correction * (a.mass_kg / total_mass)
        a.position.x += move_a
        b.position.x += move_b
        a.rect.x = int(a.position.x)
        b.rect.x = int(b.position.x)
        # Impulse-based response at contact point with friction and angular effects
        normal = pygame.Vector2(1, 0) if b.rect.centerx > a.rect.centerx else pygame.Vector2(-1, 0)
        # Contact point approx: on A's side along normal, y clamped to overlap region
        cp_y = max(min(a.rect.centery, b.rect.bottom), b.rect.top)
        cp_x = a.rect.right if normal.x > 0 else a.rect.left
        contact_point = pygame.Vector2(cp_x, cp_y)
        # Relative velocity at contact
        ra = contact_point - pygame.Vector2(a.rect.center)
        rb = contact_point - pygame.Vector2(b.rect.center)
        def cross_scalar_vec(s: float, v: pygame.Vector2) -> pygame.Vector2:
            return pygame.Vector2(-s * v.y, s * v.x)
        va = a.velocity + cross_scalar_vec(a.angular_velocity, ra)
        vb = b.velocity + cross_scalar_vec(b.angular_velocity, rb)
        rv = vb - va
        vel_along_normal = rv.dot(normal)
        if vel_along_normal < 0:
            e = max(0.0, min(PLATFORM_COLLISION_RESTITUTION, min(a.restitution, b.restitution)))
            ra_cn = ra.x * normal.y - ra.y * normal.x
            rb_cn = rb.x * normal.y - rb.y * normal.x
            inv_mass_sum = (1.0 / max(a.mass_kg, 1e-6)) + (1.0 / max(b.mass_kg, 1e-6)) + (ra_cn * ra_cn) / max(a.moment_of_inertia(), 1e-6) + (rb_cn * rb_cn) / max(b.moment_of_inertia(), 1e-6)
            j = -(1 + e) * vel_along_normal / inv_mass_sum
            impulse = normal * j
            a.velocity -= impulse / max(a.mass_kg, 1e-6)
            b.velocity += impulse / max(b.mass_kg, 1e-6)
            a.angular_velocity -= (ra.x * impulse.y - ra.y * impulse.x) / max(a.moment_of_inertia(), 1e-6)
            b.angular_velocity += (rb.x * impulse.y - rb.y * impulse.x) / max(b.moment_of_inertia(), 1e-6)
            # Friction
            tangent = (rv - normal * rv.dot(normal))
            if tangent.length_squared() > 1e-8:
                tangent = tangent.normalize()
                ra_ct = ra.x * tangent.y - ra.y * tangent.x
                rb_ct = rb.x * tangent.y - rb.y * tangent.x
                inv_mass_t = (1.0 / max(a.mass_kg, 1e-6)) + (1.0 / max(b.mass_kg, 1e-6)) + (ra_ct * ra_ct) / max(a.moment_of_inertia(), 1e-6) + (rb_ct * rb_ct) / max(b.moment_of_inertia(), 1e-6)
                jt = -rv.dot(tangent) / inv_mass_t
                mu = PLATFORM_FRICTION_COEFF
                jt = max(-mu * j, min(mu * j, jt))
                friction_impulse = tangent * jt
                a.velocity -= friction_impulse / max(a.mass_kg, 1e-6)
                b.velocity += friction_impulse / max(b.mass_kg, 1e-6)
                a.angular_velocity -= (ra.x * friction_impulse.y - ra.y * friction_impulse.x) / max(a.moment_of_inertia(), 1e-6)
                b.angular_velocity += (rb.x * friction_impulse.y - rb.y * friction_impulse.x) / max(b.moment_of_inertia(), 1e-6)
    else:
        # Separate along Y using Baumgarte positional correction
        penetration = abs(overlap_y)
        correction = (penetration - PENETRATION_SLOP)
        if correction < 0:
            correction = 0
        correction *= PENETRATION_CORRECT_PERCENT
        if penetration > DEEP_PENETRATION_THRESHOLD:
            correction += (penetration - DEEP_PENETRATION_THRESHOLD) * (0.5 + SEPARATION_BIAS)
        sign = 1 if overlap_y > 0 else -1
        move_a = -sign * correction * (b.mass_kg / total_mass)
        move_b = sign * correction * (a.mass_kg / total_mass)
        a.position.y += move_a
        b.position.y += move_b
        a.rect.y = int(a.position.y)
        b.rect.y = int(b.position.y)
        # Impulse-based response at contact point with friction and angular effects
        normal = pygame.Vector2(0, 1) if b.rect.centery > a.rect.centery else pygame.Vector2(0, -1)
        # Contact point approx: on A's side along normal, x clamped to overlap region
        cp_x = max(min(a.rect.centerx, b.rect.right), b.rect.left)
        cp_y = a.rect.bottom if normal.y > 0 else a.rect.top
        contact_point = pygame.Vector2(cp_x, cp_y)
        # Relative velocity at contact
        ra = contact_point - pygame.Vector2(a.rect.center)
        rb = contact_point - pygame.Vector2(b.rect.center)
        def cross_scalar_vec(s: float, v: pygame.Vector2) -> pygame.Vector2:
            return pygame.Vector2(-s * v.y, s * v.x)
        va = a.velocity + cross_scalar_vec(a.angular_velocity, ra)
        vb = b.velocity + cross_scalar_vec(b.angular_velocity, rb)
        rv = vb - va
        vel_along_normal = rv.dot(normal)
        if vel_along_normal < 0:
            e = max(0.0, min(PLATFORM_COLLISION_RESTITUTION, min(a.restitution, b.restitution)))
            ra_cn = ra.x * normal.y - ra.y * normal.x
            rb_cn = rb.x * normal.y - rb.y * normal.x
            inv_mass_sum = (1.0 / max(a.mass_kg, 1e-6)) + (1.0 / max(b.mass_kg, 1e-6)) + (ra_cn * ra_cn) / max(a.moment_of_inertia(), 1e-6) + (rb_cn * rb_cn) / max(b.moment_of_inertia(), 1e-6)
            j = -(1 + e) * vel_along_normal / inv_mass_sum
            impulse = normal * j
            a.velocity -= impulse / max(a.mass_kg, 1e-6)
            b.velocity += impulse / max(b.mass_kg, 1e-6)
            a.angular_velocity -= (ra.x * impulse.y - ra.y * impulse.x) / max(a.moment_of_inertia(), 1e-6)
            b.angular_velocity += (rb.x * impulse.y - rb.y * impulse.x) / max(b.moment_of_inertia(), 1e-6)
            # Friction
            tangent = (rv - normal * rv.dot(normal))
            if tangent.length_squared() > 1e-8:
                tangent = tangent.normalize()
                ra_ct = ra.x * tangent.y - ra.y * tangent.x
                rb_ct = rb.x * tangent.y - rb.y * tangent.x
                inv_mass_t = (1.0 / max(a.mass_kg, 1e-6)) + (1.0 / max(b.mass_kg, 1e-6)) + (ra_ct * ra_ct) / max(a.moment_of_inertia(), 1e-6) + (rb_ct * rb_ct) / max(b.moment_of_inertia(), 1e-6)
                jt = -rv.dot(tangent) / inv_mass_t
                mu = PLATFORM_FRICTION_COEFF
                jt = max(-mu * j, min(mu * j, jt))
                friction_impulse = tangent * jt
                a.velocity -= friction_impulse / max(a.mass_kg, 1e-6)
                b.velocity += friction_impulse / max(b.mass_kg, 1e-6)
                a.angular_velocity -= (ra.x * friction_impulse.y - ra.y * friction_impulse.x) / max(a.moment_of_inertia(), 1e-6)
                b.angular_velocity += (rb.x * friction_impulse.y - rb.y * friction_impulse.x) / max(b.moment_of_inertia(), 1e-6)


def _platform_axes(p: 'MovingPlatform') -> tuple[pygame.Vector2, pygame.Vector2, pygame.Vector2, float, float]:
    center = pygame.Vector2(p.rect.center)
    ux = pygame.Vector2(math.cos(p.angle_rad), math.sin(p.angle_rad))
    uy = pygame.Vector2(-ux.y, ux.x)
    hx = p.rect.width * 0.5
    hy = p.rect.height * 0.5
    return center, ux, uy, hx, hy


def _project_obb(center: pygame.Vector2, ux: pygame.Vector2, uy: pygame.Vector2, hx: float, hy: float, axis: pygame.Vector2) -> tuple[float, float]:
    c = center.dot(axis)
    r = hx * abs(axis.dot(ux)) + hy * abs(axis.dot(uy))
    return c - r, c + r


def _project_aabb(rect: pygame.Rect, axis: pygame.Vector2) -> tuple[float, float]:
    c = pygame.Vector2(rect.center).dot(axis)
    ex = rect.width * 0.5
    ey = rect.height * 0.5
    r = ex * abs(axis.x) + ey * abs(axis.y)
    return c - r, c + r


def _support_point(center: pygame.Vector2, ux: pygame.Vector2, uy: pygame.Vector2, hx: float, hy: float, direction: pygame.Vector2) -> pygame.Vector2:
    sx = 1.0 if direction.dot(ux) >= 0 else -1.0
    sy = 1.0 if direction.dot(uy) >= 0 else -1.0
    return center + ux * (sx * hx) + uy * (sy * hy)


def _obb_overlap_and_axis(a: 'MovingPlatform', b: 'MovingPlatform') -> tuple[bool, float, pygame.Vector2]:
    ca, ax, ay, ahx, ahy = _platform_axes(a)
    cb, bx, by, bhx, bhy = _platform_axes(b)
    axes = [ax, ay, bx, by]
    min_overlap = float('inf')
    best_axis = pygame.Vector2(1, 0)
    for axis in axes:
        axis_n = axis.normalize()
        amin, amax = _project_obb(ca, ax, ay, ahx, ahy, axis_n)
        bmin, bmax = _project_obb(cb, bx, by, bhx, bhy, axis_n)
        overlap = min(amax, bmax) - max(amin, bmin)
        if overlap <= 0:
            return False, 0.0, best_axis
        if overlap < min_overlap:
            min_overlap = overlap
            d = cb - ca
            best_axis = axis_n if d.dot(axis_n) >= 0 else -axis_n
    return True, min_overlap, best_axis


def resolve_platform_vs_platform(a: 'MovingPlatform', b: 'MovingPlatform') -> None:
    collided, overlap, normal = _obb_overlap_and_axis(a, b)
    if not collided:
        return
    # Positional correction
    penetration = overlap
    correction = max(0.0, penetration - PENETRATION_SLOP) * PENETRATION_CORRECT_PERCENT
    if penetration > DEEP_PENETRATION_THRESHOLD:
        correction += (penetration - DEEP_PENETRATION_THRESHOLD) * (0.5 + SEPARATION_BIAS)
    total_mass = max(a.mass_kg + b.mass_kg, 1e-6)
    a.position -= normal * (correction * (b.mass_kg / total_mass))
    b.position += normal * (correction * (a.mass_kg / total_mass))
    a.rect.topleft = (int(a.position.x), int(a.position.y))
    b.rect.topleft = (int(b.position.x), int(b.position.y))

    # Contact point via opposing support points
    ca, ax, ay, ahx, ahy = _platform_axes(a)
    cb, bx, by, bhx, bhy = _platform_axes(b)
    pa = _support_point(ca, ax, ay, ahx, ahy, normal)
    pb = _support_point(cb, bx, by, bhx, bhy, -normal)
    contact_point = (pa + pb) * 0.5

    ra = contact_point - pygame.Vector2(a.rect.center)
    rb = contact_point - pygame.Vector2(b.rect.center)

    def cross_scalar_vec(s: float, v: pygame.Vector2) -> pygame.Vector2:
        return pygame.Vector2(-s * v.y, s * v.x)

    va = a.velocity + cross_scalar_vec(a.angular_velocity, ra)
    vb = b.velocity + cross_scalar_vec(b.angular_velocity, rb)
    rv = vb - va
    vel_along_normal = rv.dot(normal)
    if vel_along_normal < 0:
        e = max(0.0, min(PLATFORM_COLLISION_RESTITUTION, min(a.restitution, b.restitution)))
        ra_cn = ra.x * normal.y - ra.y * normal.x
        rb_cn = rb.x * normal.y - rb.y * normal.x
        inv_mass_sum = (1.0 / max(a.mass_kg, 1e-6)) + (1.0 / max(b.mass_kg, 1e-6)) + (ra_cn * ra_cn) / max(a.moment_of_inertia(), 1e-6) + (rb_cn * rb_cn) / max(b.moment_of_inertia(), 1e-6)
        j = -(1 + e) * vel_along_normal / inv_mass_sum
        impulse = normal * j
        a.velocity -= impulse / max(a.mass_kg, 1e-6)
        b.velocity += impulse / max(b.mass_kg, 1e-6)
        a.angular_velocity -= (ra.x * impulse.y - ra.y * impulse.x) / max(a.moment_of_inertia(), 1e-6)
        b.angular_velocity += (rb.x * impulse.y - rb.y * impulse.x) / max(b.moment_of_inertia(), 1e-6)
        # Friction impulse
        tangent = rv - normal * rv.dot(normal)
        if tangent.length_squared() > 1e-8:
            tangent = tangent.normalize()
            ra_ct = ra.x * tangent.y - ra.y * tangent.x
            rb_ct = rb.x * tangent.y - rb.y * tangent.x
            inv_mass_t = (1.0 / max(a.mass_kg, 1e-6)) + (1.0 / max(b.mass_kg, 1e-6)) + (ra_ct * ra_ct) / max(a.moment_of_inertia(), 1e-6) + (rb_ct * rb_ct) / max(b.moment_of_inertia(), 1e-6)
            jt = -rv.dot(tangent) / inv_mass_t
            mu = PLATFORM_FRICTION_COEFF
            jt = max(-mu * j, min(mu * j, jt))
            friction_impulse = tangent * jt
            a.velocity -= friction_impulse / max(a.mass_kg, 1e-6)
            b.velocity += friction_impulse / max(b.mass_kg, 1e-6)
            a.angular_velocity -= (ra.x * friction_impulse.y - ra.y * friction_impulse.x) / max(a.moment_of_inertia(), 1e-6)
            b.angular_velocity += (rb.x * friction_impulse.y - rb.y * friction_impulse.x) / max(b.moment_of_inertia(), 1e-6)


def resolve_player_vs_moving_platform(player: Player, plat: 'MovingPlatform') -> bool:
    center, ux, uy, hx, hy = _platform_axes(plat)
    axes = [ux, uy, pygame.Vector2(1, 0), pygame.Vector2(0, 1)]
    smallest_overlap = float('inf')
    smallest_axis: pygame.Vector2 | None = None

    for axis in axes:
        amin, amax = _project_obb(center, ux, uy, hx, hy, axis)
        bmin, bmax = _project_aabb(player.rect, axis)
        overlap = min(amax, bmax) - max(amin, bmin)
        if overlap <= 0:
            return False
        if overlap < smallest_overlap:
            smallest_overlap = overlap
            # axis direction from platform to player
            d = (pygame.Vector2(player.rect.center) - center)
            smallest_axis = axis if d.dot(axis) >= 0 else -axis

    if smallest_axis is None:
        return False
    # Minimum translation vector to separate player out of platform
    mtv = smallest_axis * smallest_overlap
    player.position += mtv
    player.rect.topleft = (int(player.position.x), int(player.position.y))
    # Remove velocity along collision normal if moving into it; add slight restitution from platform
    n = smallest_axis.normalize()
    vn = player.velocity.dot(n)
    if vn < 0:
        player.velocity -= n * vn * (1 + plat.restitution * 0.5)
    # Grounding if normal is mostly upward
    if n.y < -0.5:
        angle_limit_rad = math.radians(PLATFORM_GROUND_ANGLE_LIMIT_DEG)
        if abs(plat.angle_rad) <= angle_limit_rad:
            player.on_ground = True
    return True


def handle_player_ground_friction(player: Player, dt: float) -> None:
    if player.on_ground:
        friction_impulse = GROUND_FRICTION * player.mass_kg
        if abs(player.velocity.x) <= friction_impulse:
            player.velocity.x = 0.0
        else:
            player.velocity.x -= math.copysign(friction_impulse, player.velocity.x)


def draw_missile(surface: pygame.Surface, missile: Missile) -> None:
    # Draw as a rotated triangle for visual heading
    center = pygame.Vector2(missile.rect.center)
    length = max(missile.rect.w, 18)
    width = max(missile.rect.h, 10)
    forward = pygame.Vector2(math.cos(missile.heading_rad), math.sin(missile.heading_rad))
    right = pygame.Vector2(-forward.y, forward.x)
    p1 = center + forward * (length * 0.6)
    p2 = center - forward * (length * 0.6) + right * (width * 0.6)
    p3 = center - forward * (length * 0.6) - right * (width * 0.6)
    pygame.draw.polygon(surface, missile.color, [p1, p2, p3])


def spawn_missile(now_s: float, player_center: pygame.Vector2) -> Missile:
    side = random.choice(["left", "right", "top", "bottom"]) 
    margin = 40
    if side == "left":
        pos = (random.randint(-margin - 100, -margin), random.randint(0, WINDOW_HEIGHT))
    elif side == "right":
        pos = (random.randint(WINDOW_WIDTH + margin, WINDOW_WIDTH + margin + 100), random.randint(0, WINDOW_HEIGHT))
    elif side == "top":
        pos = (random.randint(0, WINDOW_WIDTH), random.randint(-margin - 100, -margin))
    else:
        pos = (random.randint(0, WINDOW_WIDTH), random.randint(WINDOW_HEIGHT + margin, WINDOW_HEIGHT + margin + 100))
    delay = random.uniform(EXPLOSION_MIN_DELAY_S, EXPLOSION_MAX_DELAY_S)
    missile = Missile(pygame.Rect(pos[0], pos[1], MISSILE_WIDTH, MISSILE_HEIGHT), explode_at_s=now_s + delay)
    to_player = (player_center - pygame.Vector2(pos)).normalize() if player_center != pygame.Vector2(pos) else pygame.Vector2(1, 0)
    missile.heading_rad = math.atan2(to_player.y, to_player.x)
    return missile


class Explosion:
    def __init__(self, center: pygame.Vector2, radius: float, life_s: float) -> None:
        self.center = pygame.Vector2(center)
        self.radius = radius
        self.life_s = life_s
        self.max_life_s = life_s

    def apply_impulses(self, objects: list[GameObject], impulse_strength: float) -> None:
        for obj in objects:
            offset = pygame.Vector2(obj.rect.center) - self.center
            distance = offset.length()
            if distance <= 1e-6:
                direction = pygame.Vector2(1, 0)
                distance = 0.001
            else:
                direction = offset.normalize()
            if distance < self.radius:
                falloff = 1.0 - (distance / self.radius)
                # Add tangential component to encourage spin
                radial = direction
                tangential = pygame.Vector2(-radial.y, radial.x)
                impulse_dir = (radial * (1.0 - EXPLOSION_TANGENTIAL_FRACTION) + tangential * EXPLOSION_TANGENTIAL_FRACTION).normalize()
                impulse = impulse_dir * (impulse_strength * falloff)
                obj.apply_impulse(impulse)
                # Apply torque proportional to perpendicular lever arm
                # r x F magnitude for 2D about center is |r|*|F|*sin(theta) = cross(r_hat, F)
                # Use sign based on 2D cross product z-component
                r = offset
                F = impulse
                torque = r.x * F.y - r.y * F.x
                # Scale torque by object size to have noticeable spin on platforms
                size_factor = max(1.0, (obj.rect.width + obj.rect.height) / 200.0)
                obj.apply_angular_impulse(torque * EXPLOSION_TORQUE_FACTOR * size_factor)

    def draw(self, surface: pygame.Surface) -> None:
        t = 1.0 - max(0.0, min(1.0, self.life_s / self.max_life_s))
        # Expand ring and fade
        radius = int(self.radius * (0.8 + 0.4 * t))
        alpha = max(0, min(255, int(220 * (1.0 - t))))
        gfx = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(gfx, (255, 230, 120, alpha), (radius, radius), radius, width=max(2, int(8 * (1.0 - t))))
        surface.blit(gfx, (self.center.x - radius, self.center.y - radius))


class Shield:
    def __init__(self, owner: 'Player', direction: pygame.Vector2, length: float, width: float, life_s: float):
        self.owner = owner
        self.direction = direction.normalize() if direction.length_squared() > 0 else pygame.Vector2(1, 0)
        self.length = length
        self.width = width
        self.life_s = life_s
        self.max_life_s = life_s

    @property
    def rect(self) -> pygame.Rect:
        # Create a rectangle protruding from the player's center in the direction
        center = pygame.Vector2(self.owner.rect.center)
        dir_vec = self.direction
        # Compute rectangle corners axis-aligned by constructing a temporary surface oriented by direction
        # For simplicity, we return an axis-aligned bounding rect that approximates the slab
        end = center + dir_vec * self.length
        if abs(dir_vec.x) > abs(dir_vec.y):
            # Horizontal shield
            w = int(self.length)
            h = int(self.width)
            if dir_vec.x >= 0:
                return pygame.Rect(int(center.x), int(center.y - h // 2), w, h)
            else:
                return pygame.Rect(int(center.x - w), int(center.y - h // 2), w, h)
        else:
            # Vertical shield
            w = int(self.width)
            h = int(self.length)
            if dir_vec.y >= 0:
                return pygame.Rect(int(center.x - w // 2), int(center.y), w, h)
            else:
                return pygame.Rect(int(center.x - w // 2), int(center.y - h), w, h)

    def draw(self, surface: pygame.Surface) -> None:
        frac = max(0.0, min(1.0, self.life_s / self.max_life_s))
        color = (120, 210, 255, int(140 * frac))
        gfx = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(gfx, color, pygame.Rect(0, 0, self.rect.width, self.rect.height), border_radius=6)
        surface.blit(gfx, (self.rect.x, self.rect.y))


def format_time(seconds: float) -> str:
    whole = int(seconds)
    ms = int((seconds - whole) * 1000)
    m = whole // 60
    s = whole % 60
    return f"{m:02d}:{s:02d}.{ms:03d}"


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Runner Man - Survive the Homing Missiles")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    def new_game_state():
        player = Player(pygame.Rect(120, WINDOW_HEIGHT - 160, PLAYER_WIDTH, PLAYER_HEIGHT))
        # Stage: ground and a few floating platforms. They have mass but are anchored (fixed in place).
        ground = Platform(pygame.Rect(0, WINDOW_HEIGHT - 40, WINDOW_WIDTH, 40), Color(40, 40, 40), mass_kg=10000.0, is_anchored=True)
        moving_platforms: list[MovingPlatform] = [
            MovingPlatform(pygame.Rect(220, WINDOW_HEIGHT - 240, 160, 22), Color(60, 100, 160), mass_kg=500.0, velocity=pygame.Vector2(140, 0)),
            MovingPlatform(pygame.Rect(520, WINDOW_HEIGHT - 320, 160, 22), Color(100, 160, 60), mass_kg=500.0, velocity=pygame.Vector2(-120, -20)),
            MovingPlatform(pygame.Rect(760, WINDOW_HEIGHT - 380, 90, 22), Color(160, 100, 60), mass_kg=480.0, velocity=pygame.Vector2(-160, 0)),
        ]
        platforms: list[Platform] = [ground, *moving_platforms]
        missiles: list[Missile] = []
        explosions: list[Explosion] = []
        spawn_timer = 0.0
        spawn_interval = MISSILE_SPAWN_INTERVAL_START_S
        start_ticks = pygame.time.get_ticks()
        time_since_grounded = 0.0
        jump_buffer_timer = 0.0
        shields: list[Shield] = []
        return player, platforms, missiles, explosions, shields, spawn_timer, spawn_interval, start_ticks, time_since_grounded, jump_buffer_timer

    player, platforms, missiles, explosions, shields, spawn_timer, spawn_interval, start_ticks, time_since_grounded, jump_buffer_timer = new_game_state()
    running = True
    game_over = False
    survived_seconds = 0.0
    difficulty_elapsed = 0.0

    while running:
        dt_ms = clock.tick(FPS)
        dt = dt_ms / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if not game_over and event.key in (pygame.K_UP, pygame.K_w, pygame.K_SPACE):
                    jump_buffer_timer = JUMP_BUFFER_S
                if game_over and event.key == pygame.K_r:
                    player, platforms, missiles, explosions, shields, spawn_timer, spawn_interval, start_ticks, time_since_grounded, jump_buffer_timer = new_game_state()
                    game_over = False
                # Activate shield on Left Shift or Right Shift
                if not game_over and event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                    if player.shield_cooldown_s <= 0.0:
                        # Determine direction from held keys, fallback to last facing
                        dir_vec = pygame.Vector2(0, 0)
                        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                            dir_vec.x -= 1
                        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                            dir_vec.x += 1
                        if keys[pygame.K_UP] or keys[pygame.K_w]:
                            dir_vec.y -= 1
                        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                            dir_vec.y += 1
                        if dir_vec.length_squared() == 0:
                            dir_vec = player.facing.copy()
                        shield = Shield(owner=player, direction=dir_vec, length=120.0, width=34.0, life_s=0.18)
                        shields.append(shield)
                        player.shield_cooldown_s = 0.9
            elif event.type == pygame.KEYUP:
                # Variable jump height: if jump released while moving upward, cut vertical speed
                if event.key in (pygame.K_UP, pygame.K_w, pygame.K_SPACE):
                    if player.velocity.y < -JUMP_CUTOFF_SPEED:
                        player.velocity.y = -JUMP_CUTOFF_SPEED

        keys = pygame.key.get_pressed()
        if not game_over:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                player.move_left(dt)
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                player.move_right(dt)
            # Update facing from vertical inputs if dominant
            vertical = 0
            horizontal = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                vertical -= 1
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                vertical += 1
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                horizontal -= 1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                horizontal += 1
            if abs(vertical) > abs(horizontal) and vertical != 0:
                player.facing.update(0, vertical)

        # Physics update
        if not game_over:
            now_s = (pygame.time.get_ticks() - start_ticks) / 1000.0
            # Early consume jump buffer using coyote time
            if jump_buffer_timer > 0.0 and time_since_grounded <= COYOTE_TIME_S:
                player.velocity.y = -PLAYER_JUMP_SPEED
                player.on_ground = False
                jump_buffer_timer = 0.0

            player.on_ground = False
            player.apply_gravity()
            player.integrate(dt)
            # Constrain to world horizontally
            if player.rect.left < 0:
                player.position.x = 0
                player.rect.left = 0
                player.velocity.x = 0
            elif player.rect.right > WINDOW_WIDTH:
                player.position.x = WINDOW_WIDTH - player.rect.width
                player.rect.right = WINDOW_WIDTH
                player.velocity.x = 0

            # Update platforms (integrate, bounce on bounds)
            for p in platforms:
                if isinstance(p, MovingPlatform):
                    # Moving platforms ignore gravity; just move and bounce
                    p.integrate(dt)
                    p.update_motion_and_bounce(dt)

            # Collide moving platforms with ground
            if platforms:
                ground = platforms[0]
                for p in platforms:
                    if isinstance(p, MovingPlatform):
                        aabb_resolve_dynamic_static(p, ground)

            # Collide platforms with each other using OBB SAT + impulse, multi-iteration
            dynamic_platforms = [p for p in platforms if isinstance(p, MovingPlatform)]
            for _ in range(SOLVER_ITERATIONS):
                for i in range(len(dynamic_platforms)):
                    for j in range(i + 1, len(dynamic_platforms)):
                        resolve_platform_vs_platform(dynamic_platforms[i], dynamic_platforms[j])

            # Collisions with platforms for player
            for p in platforms:
                if isinstance(p, MovingPlatform):
                    if resolve_player_vs_moving_platform(player, p):
                        continue
                aabb_resolve_dynamic_static(player, p)
            # Ground probe to robustly detect standing even on edges
            grounded_probe = False
            if player.velocity.y >= -1.0:  # allow slight upward motion tolerance
                probe = player.rect.copy()
                probe.y += 2
                probe.h += 2
                for p in platforms:
                    if probe.colliderect(p.rect):
                        if isinstance(p, MovingPlatform):
                            angle_limit_rad = math.radians(PLATFORM_GROUND_ANGLE_LIMIT_DEG)
                            if abs(p.angle_rad) <= angle_limit_rad:
                                grounded_probe = True
                                break
                        else:
                            grounded_probe = True
                            break
            player.on_ground = player.on_ground or grounded_probe
            # Update coyote timer
            if player.on_ground:
                time_since_grounded = 0.0
            else:
                time_since_grounded += dt

            handle_player_ground_friction(player, dt)
            # Decay jump buffer timer
            jump_buffer_timer = max(0.0, jump_buffer_timer - dt)
            # Decay invulnerability and shield cooldown
            if player.invuln_timer_s > 0.0:
                player.invuln_timer_s = max(0.0, player.invuln_timer_s - dt)
            if player.shield_cooldown_s > 0.0:
                player.shield_cooldown_s = max(0.0, player.shield_cooldown_s - dt)

            # Update missiles
            for m in list(missiles):
                m.apply_gravity()
                m.update_heading_towards(pygame.Vector2(player.rect.center), dt)
                m.apply_thrust()
                m.integrate(dt)
                m.clamp_speed()
                # Collide missiles with platforms
                for p in platforms:
                    aabb_resolve_dynamic_static(m, p)
                # Check hit player
                if m.rect.colliderect(player.rect):
                    # Missile causes an immediate explosion on impact as well
                    exp = Explosion(pygame.Vector2(m.rect.center), EXPLOSION_RADIUS, EXPLOSION_LIFE_S)
                    explosions.append(exp)
                    affected: list[GameObject] = [player, *platforms]
                    exp.apply_impulses(affected, EXPLOSION_IMPULSE)
                    # Missile is consumed; player takes a hit
                    missiles.remove(m)
                    # Stabilize after an impact-triggered blast too
                    dynamic_platforms = [p for p in platforms if isinstance(p, MovingPlatform)]
                    for _ in range(STABILIZE_AFTER_EXPLOSION_PASSES):
                        for i in range(len(dynamic_platforms)):
                            for j in range(i + 1, len(dynamic_platforms)):
                                resolve_platform_vs_platform(dynamic_platforms[i], dynamic_platforms[j])
                    if player.invuln_timer_s <= 0.0:
                        player.take_hit()
                        if player.lives <= 0:
                            game_over = True
                            survived_seconds = (pygame.time.get_ticks() - start_ticks) / 1000.0
                # Timed explosion
                if now_s >= m.explode_at_s:
                    # Create explosion and apply impulses
                    exp = Explosion(pygame.Vector2(m.rect.center), EXPLOSION_RADIUS, EXPLOSION_LIFE_S)
                    explosions.append(exp)
                    # Objects affected: player and all platforms (dynamic only will react)
                    affected: list[GameObject] = [player, *platforms]
                    exp.apply_impulses(affected, EXPLOSION_IMPULSE)
                    missiles.remove(m)
                    # Immediately run a few extra platform-platform solver passes to de-overlap after blasts
                    dynamic_platforms = [p for p in platforms if isinstance(p, MovingPlatform)]
                    for _ in range(STABILIZE_AFTER_EXPLOSION_PASSES):
                        for i in range(len(dynamic_platforms)):
                            for j in range(i + 1, len(dynamic_platforms)):
                                resolve_platform_vs_platform(dynamic_platforms[i], dynamic_platforms[j])

            # Spawn missiles over time with increasing frequency
            spawn_timer += dt
            difficulty_elapsed += dt
            # Ease spawn interval towards minimum over minutes
            target_interval = max(
                MISSILE_SPAWN_INTERVAL_MIN_S,
                MISSILE_SPAWN_INTERVAL_START_S - (difficulty_elapsed / 60.0) * MISSILE_SPAWN_INTERVAL_DECAY_PER_MIN,
            )
            # Smoothly approach target interval
            spawn_interval = spawn_interval + (target_interval - spawn_interval) * min(1.0, dt * 0.5)
            if spawn_timer >= spawn_interval:
                spawn_timer = 0.0
                # Scale missile count with difficulty time
                difficulty_factor = max(1, int(1 + difficulty_elapsed / 20.0))
                for _ in range(difficulty_factor):
                    missiles.append(spawn_missile(now_s, pygame.Vector2(player.rect.center)))

            # Update and age explosions
            for exp in list(explosions):
                exp.life_s -= dt
                if exp.life_s <= 0:
                    explosions.remove(exp)

            # Update shields
            for sh in list(shields):
                sh.life_s -= dt
                if sh.life_s <= 0:
                    shields.remove(sh)
                else:
                    # Shield collides with missiles
                    for m in list(missiles):
                        if sh.rect.colliderect(m.rect):
                            # explode missile
                            exp = Explosion(pygame.Vector2(m.rect.center), EXPLOSION_RADIUS, EXPLOSION_LIFE_S)
                            explosions.append(exp)
                            affected: list[GameObject] = [player, *platforms]
                            exp.apply_impulses(affected, EXPLOSION_IMPULSE)
                            missiles.remove(m)
                    # Shield pushes platforms
                    for p in platforms:
                        if isinstance(p, MovingPlatform) and sh.rect.colliderect(p.rect):
                            # Apply a directional impulse from player towards shield direction
                            center = pygame.Vector2(p.rect.center)
                            radial = (center - pygame.Vector2(player.rect.center))
                            if radial.length_squared() < 1e-6:
                                radial = sh.direction
                            dir_vec = sh.direction.normalize()
                            impulse = dir_vec * SHIELD_PUSH_IMPULSE
                            p.apply_impulse(impulse)
                            # Add torque based on lever arm
                            torque = (radial.x * impulse.y - radial.y * impulse.x) * SHIELD_TORQUE_FACTOR
                            p.apply_angular_impulse(torque)

        # Drawing
        screen.fill(Color(22, 24, 28))
        # Platforms
        for p in platforms:
            if isinstance(p, MovingPlatform) and abs(p.angle_rad) > 1e-3:
                # Draw rotated platform
                surf = pygame.Surface((p.rect.width, p.rect.height), pygame.SRCALPHA)
                pygame.draw.rect(surf, p.color, pygame.Rect(0, 0, p.rect.width, p.rect.height))
                rotated = pygame.transform.rotate(surf, -math.degrees(p.angle_rad))
                rect = rotated.get_rect(center=p.rect.center)
                screen.blit(rotated, rect.topleft)
            else:
                pygame.draw.rect(screen, p.color, p.rect)
        # Player
        pygame.draw.rect(screen, player.color, player.rect, border_radius=6)
        # Missiles
        for m in missiles:
            draw_missile(screen, m)
        # Explosions
        for exp in explosions:
            exp.draw(screen)
        # Shields
        for sh in shields:
            sh.draw(screen)

        # HUD
        now_s = (pygame.time.get_ticks() - start_ticks) / 1000.0
        elapsed_s = survived_seconds if game_over else now_s
        timer_surf = font.render(f"Survival: {format_time(elapsed_s)}", True, Color(230, 230, 230))
        screen.blit(timer_surf, (12, 10))
        lives_surf = font.render(f"Lives: {player.lives}", True, Color(230, 200, 200))
        screen.blit(lives_surf, (12, 34))
        info_surf = font.render("Move: A/D or Left/Right  Jump: W/Up/Space  Quit: Esc  Restart: R", True, Color(160, 170, 180))
        screen.blit(info_surf, (12, 58))

        if game_over:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))
            go1 = font.render("Game Over!", True, Color(255, 200, 200))
            go2 = font.render(f"You survived {format_time(survived_seconds)}", True, Color(255, 255, 255))
            go3 = font.render("Press R to restart", True, Color(200, 220, 255))
            screen.blit(go1, (WINDOW_WIDTH // 2 - go1.get_width() // 2, WINDOW_HEIGHT // 2 - 30))
            screen.blit(go2, (WINDOW_WIDTH // 2 - go2.get_width() // 2, WINDOW_HEIGHT // 2))
            screen.blit(go3, (WINDOW_WIDTH // 2 - go3.get_width() // 2, WINDOW_HEIGHT // 2 + 30))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()

