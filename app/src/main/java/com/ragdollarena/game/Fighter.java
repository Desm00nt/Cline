package com.ragdollarena.game;

/** Runtime state of one little person: stats, active pose and ragdoll body. */
public class Fighter {

    public final Chars.Data data;
    public final Ragdoll body;
    public final int team; // 0 = player, 1 = rivals

    public float maxHp, hp, power, mass, speed, dashCd;
    public float dashTimer, dashActive, hitFlash, stagger, koTimer, animPhase;
    public boolean ko, out, eliminated;
    public int facing;
    public float walkDir;     // desired horizontal movement, -1 / 0 / 1
    public float aiTimer;
    public float alpha = 1f;
    public float idleSeed;
    public float lastDamage;
    public float bobScale = 1f;
    public float dmgCd;
    public boolean showHp = true;

    public Fighter(Chars.Data data, int team, float x, float groundY,
                   float hpMult, float powerMult, float massMult, float speedMult,
                   float startHpFraction) {
        this.data = data;
        this.team = team;
        this.facing = team == 0 ? 1 : -1;
        this.maxHp = data.hp * hpMult;
        this.hp = Math.max(1f, maxHp * startHpFraction);
        this.power = data.power * powerMult;
        this.mass = Math.max(0.5f, data.mass * massMult);
        this.speed = data.speed * speedMult;
        this.dashCd = data.dashCd * (data.perk == Chars.PERK_CD ? 0.7f : 1f);
        this.body = new Ragdoll(x, groundY - 51f, facing);
        this.aiTimer = 0.6f + (float) Math.random() * 1.4f;
        this.idleSeed = (float) (Math.random() * 6.28);
        for (int i = 0; i < Ragdoll.N; i++) {
            body.setTarget(i, body.x[i], body.y[i]);
        }
    }

    public boolean canDash() {
        return !ko && !out && !eliminated && dashTimer <= 0f && stagger <= 0f;
    }

    public void dash(float targetX, float targetY, float strength) {
        if (!canDash()) return;
        float dirX = targetX - body.x[Ragdoll.CHEST];
        float dirY = targetY - body.y[Ragdoll.CHEST];
        float len = (float) Math.sqrt(dirX * dirX + dirY * dirY);
        if (len < 1f) {
            dirX = facing;
            dirY = 0f;
        } else {
            dirX /= len;
            dirY /= len;
        }
        if (Math.abs(dirX) > 0.12f) facing = dirX > 0 ? 1 : -1;
        float v = strength / (float) Math.sqrt(mass) * (0.75f + speed * 0.25f);
        float jump = -150f - Math.max(0f, -dirY) * 140f;
        body.applyImpulse(Ragdoll.PELVIS, dirX * v, dirY * v * 0.6f + jump, 1f / 60f);
        body.applyImpulse(Ragdoll.CHEST, dirX * v * 0.55f, dirY * v * 0.35f + jump * 0.5f, 1f / 60f);
        body.applyImpulse(Ragdoll.HAND_L, dirX * v * 0.4f, dirY * v * 0.2f, 1f / 60f);
        body.applyImpulse(Ragdoll.HAND_R, dirX * v * 0.4f, dirY * v * 0.2f, 1f / 60f);
        dashActive = 0.30f;
        dashTimer = dashCd;
        stagger = 0f;
        animPhase = 0f;
    }

    /** Returns true if this hit knocked the fighter out. */
    public boolean hurt(float dmg) {
        if (ko) return false;
        float taken = dmg * (data.perk == Chars.PERK_ARMOR ? 0.75f : 1f);
        hp -= taken;
        lastDamage = taken;
        hitFlash = 1f;
        stagger = Math.max(stagger, 0.42f);
        dashActive = 0f;
        if (hp <= 0f) {
            hp = 0f;
            ko = true;
            koTimer = 0f;
            stagger = 0f;
            return true;
        }
        return false;
    }

    public void update(float dt, float groundY, float platL, float platR) {
        if (dashTimer > 0f) dashTimer -= dt;
        if (dashActive > 0f) dashActive -= dt;
        if (hitFlash > 0f) hitFlash = Math.max(0f, hitFlash - dt * 4f);
        if (stagger > 0f && !ko) stagger = Math.max(0f, stagger - dt);
        if (ko) {
            koTimer += dt;
            for (int i = 0; i < Ragdoll.N; i++) body.spring[i] = 0f;
            if (!out && koTimer > 2.4f) eliminated = true;
            if (eliminated) alpha = Math.max(0f, alpha - dt * 1.3f);
            body.step(dt, 1100f, 0f, groundY, platL, platR, 0.5f);
            if (body.y[Ragdoll.PELVIS] > groundY + 110f) out = true;
            if (out) alpha = Math.max(0f, alpha - dt * 1.6f);
            return;
        }

        float px = body.x[Ragdoll.PELVIS];
        float py = body.y[Ragdoll.PELVIS];
        boolean onPlat = px > platL - 14f && px < platR + 14f;
        boolean grounded = onPlat && py > groundY - 105f && py < groundY - 8f
                && (body.y[Ragdoll.FOOT_L] > groundY - 26f || body.y[Ragdoll.FOOT_R] > groundY - 26f)
                && py < body.y[Ragdoll.FOOT_L] - 24f;
        if (dashActive > 0.12f && body.vy(Ragdoll.PELVIS, dt) < -80f) grounded = false;

        animPhase += dt * (5.5f + Math.abs(walkDir) * 5.5f);
        float phase = animPhase + idleSeed;
        float bob = (float) Math.sin(phase) * 1.8f * bobScale;
        float lean = 0f;
        float moving = 0f;
        if (grounded) {
            if (dashActive > 0f) {
                lean = 13f * facing;
                moving = 1.2f;
            } else if (Math.abs(walkDir) > 0.1f) {
                lean = 6f * facing;
                moving = 1f;
            }
        }
        float rootY = grounded ? groundY - 51f + bob * 0.35f : py;
        float rootX = px + lean * 0.25f;
        body.setTarget(Ragdoll.PELVIS, px, rootY);
        body.setTarget(Ragdoll.CHEST, rootX + lean, rootY - 38f + bob * 0.5f);
        body.setTarget(Ragdoll.HEAD, rootX + lean * 1.3f, rootY - 58f + bob * 0.8f);

        float step = (float) Math.sin(phase) * 13f * moving;
        float step2 = (float) Math.cos(phase) * 13f * moving;
        float liftL = Math.max(0f, (float) Math.sin(phase)) * 8f * moving;
        float liftR = Math.max(0f, -(float) Math.sin(phase)) * 8f * moving;
        float footY = grounded ? groundY - 6.5f : body.y[Ragdoll.FOOT_L];
        float footY2 = grounded ? groundY - 6.5f : body.y[Ragdoll.FOOT_R];

        float f = facing;
        float footLX = grounded ? px - 11f * f + step * f : body.x[Ragdoll.FOOT_L];
        float footRX = grounded ? px + 11f * f - step * f : body.x[Ragdoll.FOOT_R];
        float footLY = grounded ? footY - liftL : footY;
        float footRY = grounded ? footY2 - liftR : footY2;
        body.setTarget(Ragdoll.FOOT_L, footLX, footLY);
        body.setTarget(Ragdoll.FOOT_R, footRX, footRY);
        body.setTarget(Ragdoll.KNEE_L, (px + footLX) * 0.5f - f * (2f + step * 0.3f),
                (rootY + footLY) * 0.5f - 2f);
        body.setTarget(Ragdoll.KNEE_R, (px + footRX) * 0.5f + f * (2f + step * 0.3f),
                (rootY + footRY) * 0.5f - 2f);

        float handLY, handRY, handLX, handRX;
        if (dashActive > 0f) {
            handLX = px + f * 30f;
            handRX = px + f * 22f;
            handLY = rootY - 48f;
            handRY = rootY - 32f;
        } else {
            handLX = px - 21f * f + step2 * 6f * f;
            handRX = px + 21f * f - step2 * 6f * f;
            handLY = rootY - 10f + bob * 0.5f;
            handRY = rootY - 10f + bob * 0.5f;
        }
        body.setTarget(Ragdoll.HAND_L, handLX, handLY);
        body.setTarget(Ragdoll.HAND_R, handRX, handRY);
        body.setTarget(Ragdoll.ELBOW_L, (px + handLX) * 0.5f - 9f * f, (rootY - 33f + handLY) * 0.5f);
        body.setTarget(Ragdoll.ELBOW_R, (px + handRX) * 0.5f + 9f * f, (rootY - 33f + handRY) * 0.5f);

        float stiffness = stagger > 0f ? 0.20f : 1f;
        body.step(dt, 1100f, stiffness, groundY, platL, platR, grounded ? 0.55f : 0.15f);

        if (grounded && Math.abs(walkDir) > 0.1f && dashActive <= 0f && stagger <= 0f) {
            float dx = walkDir * speed * 95f * dt;
            for (int i = 0; i < Ragdoll.N; i++) {
                body.x[i] += dx;
                body.ox[i] += dx;
                body.tx[i] += dx;
            }
        }
        if (grounded) {
            for (int gi = 0; gi < Ragdoll.N; gi++) {
                float hvx = body.vx(gi, dt);
                body.applyImpulse(gi, -hvx * 0.06f, 0f, dt);
            }
        }

        if (body.y[Ragdoll.PELVIS] > groundY + 110f) out = true;
        if (out) alpha = Math.max(0f, alpha - dt * 1.6f);

        setSprings(stiffness);
    }

    private void setSprings(float stiffness) {
        float[] base = {240f, 240f, 240f, 110f, 60f, 110f, 60f, 170f, 210f, 170f, 210f};
        for (int i = 0; i < Ragdoll.N; i++) {
            body.spring[i] = base[i] * stiffness;
        }
    }

    public float centerX() {
        return body.x[Ragdoll.PELVIS];
    }

    public float centerY() {
        return body.y[Ragdoll.CHEST];
    }
}
