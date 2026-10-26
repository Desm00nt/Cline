package com.ragdollarena.game;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Pure Java battle simulation. No Android dependencies, which means it can be
 * exercised by the headless test harness in tools/.
 */
public class BattleSim {

    public interface Listener {
        void onHit(float strength);
        void onKO();
        void onDash();
    }

    public static class Spark {
        public float x, y, vx, vy, life, maxLife, size;
        public int color;
    }

    public static class Ring {
        public float x, y, r, maxR, life, maxLife;
        public int color;
    }

    public static class Popup {
        public float x, y, life, maxLife;
        public String text;
        public int color;
        public float size;
    }

    public final float w, h;
    public final float groundY, basePlatL, basePlatR;
    public float platL, platR;
    public final ArrayList<Fighter> fighters = new ArrayList<>();
    public final ArrayList<Spark> sparks = new ArrayList<>();
    public final ArrayList<Ring> rings = new ArrayList<>();
    public final ArrayList<Popup> popups = new ArrayList<>();

    public float countdown = 2.6f;
    public float time;
    public int result; // 0 ongoing, 1 player wins, 2 rivals win
    public float endDelay;
    public float shake;
    public float slowmo;
    public boolean timeUp;
    public String resultReason = "";
    public Listener listener;
    public final Random rng = new Random(7);
    private final float[][] pairCd;
    public int totalKos;
    public int hitCount;
    public float maxImpact;

    public BattleSim(float w, float h, List<Fighter> players, List<Fighter> enemies) {
        this.w = w;
        this.h = h;
        this.groundY = h * 0.72f;
        this.basePlatL = w * 0.5f - Math.min(330f, w * 0.22f);
        this.basePlatR = w * 0.5f + Math.min(330f, w * 0.22f);
        this.platL = basePlatL;
        this.platR = basePlatR;
        fighters.addAll(players);
        fighters.addAll(enemies);
        pairCd = new float[fighters.size()][fighters.size()];
        for (Fighter f : fighters) {
            f.walkDir = 0;
        }
    }

    public boolean fighting() {
        return countdown <= 0f && result == 0;
    }

    public List<Fighter> team(int team) {
        List<Fighter> list = new ArrayList<>();
        for (Fighter f : fighters) if (f.team == team) list.add(f);
        return list;
    }

    public void update(float dt) {
        if (dt > 0.05f) dt = 0.05f;
        if (slowmo > 0f) slowmo -= dt;
        shake = Math.max(0f, shake - dt * 30f);
        for (int i = 0; i < fighters.size(); i++) {
            for (int j = 0; j < fighters.size(); j++) {
                if (pairCd[i][j] > 0f) pairCd[i][j] -= dt;
            }
        }
        if (countdown > 0f) {
            countdown -= dt;
            if (countdown <= 0f) {
                countdown = 0f;
                addPopup(w * 0.5f, groundY - 170f, "БОЙ!", 0xFFF6D365, 58f);
                shake = 10f;
            }
        }
        if (countdown > 0f) {
            for (Fighter f : fighters) {
                f.walkDir = 0;
                f.update(dt, groundY, platL, platR);
            }
            updateEffects(dt);
            return;
        }

        if (result == 0) {
            time += dt;
            if (time > 15f) {
                float shrink = Math.min(145f, (time - 15f) * 9f);
                if (time > 30f) shrink += Math.min(120f, (time - 30f) * 12f);
                platL = basePlatL + shrink;
                platR = basePlatR - shrink;
            }
            if (time > 38f) {
                for (Fighter f : fighters) {
                    if (f.ko || f.out) continue;
                    f.hp -= 20f * dt;
                    if (f.hp <= 0f) {
                        f.hp = 0f;
                        f.ko = true;
                        f.koTimer = 0f;
                        f.dashActive = 0f;
                    }
                }
            }
            updateAI(dt);
            for (Fighter f : fighters) {
                faceOpponent(f);
                f.update(dt, groundY, platL, platR);
            }
            resolveCollisions(dt);
            checkResult();
            if (time > 45f && result == 0) {
                timeUp = true;
                int playerScore = 0, enemyScore = 0;
                for (Fighter f : fighters) {
                    int score = Math.round(f.hp);
                    if (f.team == 0) playerScore += score;
                    else enemyScore += score;
                }
                finish(playerScore >= enemyScore ? 1 : 2, "ВРЕМЯ ВЫШЛО");
            }
        } else {
            endDelay += dt;
            for (Fighter f : fighters) {
                if (!f.ko) f.walkDir = 0;
                f.update(dt, groundY, platL, platR);
            }
        }
        updateEffects(dt);
    }

    private void faceOpponent(Fighter f) {
        if (f.ko || f.out) return;
        Fighter nearest = nearestEnemy(f);
        if (nearest != null) {
            float dx = nearest.centerX() - f.centerX();
            if (Math.abs(dx) > 24f) f.facing = dx > 0 ? 1 : -1;
        }
    }

    private Fighter nearestEnemy(Fighter f) {
        Fighter best = null;
        float bestD = Float.MAX_VALUE;
        for (Fighter o : fighters) {
            if (o.team == f.team || o.eliminated || o.out) continue;
            float d = Math.abs(o.centerX() - f.centerX());
            if (d < bestD) {
                bestD = d;
                best = o;
            }
        }
        return best;
    }

    private void updateAI(float dt) {
        for (Fighter f : fighters) {
            if (f.ko || f.out || f.eliminated) continue;
            Fighter target = nearestEnemy(f);
            if (target == null) continue;
            float dx = target.centerX() - f.centerX();
            float dist = Math.abs(dx);
            int teamIdx = 0;
            for (Fighter o : fighters) {
                if (o == f) break;
                if (o.team == f.team) teamIdx++;
            }
            if (f.team == 0) {
                // your squad holds the line, the player launches the attacks
                f.walkDir = 0f;
            } else {
                float standoff = 88f + teamIdx * 30f;
                f.walkDir = dist > standoff ? (dx > 0 ? 1 : -1) : 0f;
            }
            if (f.team == 1) {
                f.aiTimer -= dt;
                if (f.aiTimer <= 0f) {
                    f.aiTimer = 0.8f + rng.nextFloat() * 1.1f;
                    if (dist < 380f && f.canDash() && rng.nextFloat() < 0.78f) {
                        float aimX = target.centerX() + rng.nextFloat() * 30f - 15f;
                        float aimY = target.centerY() + 20f + rng.nextFloat() * 20f;
                        f.dash(aimX, aimY, 820f);
                        if (listener != null) listener.onDash();
                    }
                }
            }
        }
    }

    private void resolveCollisions(float dt) {
        int n = fighters.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Fighter a = fighters.get(i);
                Fighter b = fighters.get(j);
                if (a.out || b.out) continue;
                collidePair(a, b, dt, i, j);
            }
        }
    }

    private void collidePair(Fighter a, Fighter b, float dt, int ia, int ib) {
        int[] pa = {Ragdoll.CHEST, Ragdoll.PELVIS};
        int[] pb = {Ragdoll.CHEST, Ragdoll.PELVIS};
        float[] ra = {16f, 13f};
        float[] rb = {16f, 13f};
        for (int ca = 0; ca < 2; ca++) {
            for (int cb = 0; cb < 2; cb++) {
                float ax = a.body.x[pa[ca]], ay = a.body.y[pa[ca]];
                float bx = b.body.x[pb[cb]], by = b.body.y[pb[cb]];
                float dx = bx - ax, dy = by - ay;
                float d2 = dx * dx + dy * dy;
                float rsum = ra[ca] + rb[cb];
                if (d2 > rsum * rsum || d2 < 0.0001f) continue;
                float dist = (float) Math.sqrt(d2);
                float nx = dx / dist, ny = dy / dist;
                float overlap = rsum - dist;
                float ma = a.mass, mb = b.mass;
                float wa = mb / (ma + mb), wb = ma / (ma + mb);
                a.body.x[pa[ca]] -= nx * overlap * wa;
                a.body.y[pa[ca]] -= ny * overlap * wa;
                b.body.x[pb[cb]] += nx * overlap * wb;
                b.body.y[pb[cb]] += ny * overlap * wb;

                float vax = a.body.vx(pa[ca], dt), vay = a.body.vy(pa[ca], dt);
                float vbx = b.body.vx(pb[cb], dt), vby = b.body.vy(pb[cb], dt);
                float vrel = (vbx - vax) * nx + (vby - vay) * ny;
                float impact = Math.max(0f, -vrel);
                if (vrel < 0f) {
                    float jimp = -1.55f * vrel / (1f / ma + 1f / mb);
                    a.body.applyImpulse(pa[ca], -nx * jimp / ma, -ny * jimp / ma, dt);
                    b.body.applyImpulse(pb[cb], nx * jimp / mb, ny * jimp / mb, dt);
                }
                if (a.team != b.team && impact > 70f && pairCd[ia][ib] <= 0f) {
                    pairCd[ia][ib] = pairCd[ib][ia] = 0.45f;
                    hitCount++;
                    maxImpact = Math.max(maxImpact, impact);
                    float base = 12f + (impact - 70f) * 0.10f;
                    float dmgToA = base * (b.dashActive > 0f ? 2.2f : 0.5f)
                            * b.power / (float) Math.sqrt(a.mass);
                    float dmgToB = base * (a.dashActive > 0f ? 2.2f : 0.5f)
                            * a.power / (float) Math.sqrt(b.mass);
                    if (dmgToA > 1f) hurtFighter(a, dmgToA, ax, ay, b);
                    if (dmgToB > 1f) hurtFighter(b, dmgToB, bx, by, a);
                    shake = Math.min(20f, shake + impact * 0.018f);
                    if (impact > 430f) slowmo = 0.06f;
                    if (listener != null) listener.onHit(impact);
                }
            }
        }
    }

    private void hurtFighter(Fighter f, float dmg, float x, float y, Fighter attacker) {
        if (f.ko || f.out) return;
        boolean ko = f.hurt(dmg);
        addPopup(x, y - 42f, "-" + Math.round(dmg),
                attacker.dashActive > 0f ? 0xFFFFD166 : 0xFFFFFFFF, 26f);
        for (int s = 0; s < 9; s++) {
            Spark sp = new Spark();
            sp.x = x;
            sp.y = y;
            float ang = rng.nextFloat() * 6.283f;
            float sp2 = 80f + rng.nextFloat() * 260f;
            sp.vx = (float) Math.cos(ang) * sp2 + (x - attacker.centerX()) * 0.6f;
            sp.vy = (float) Math.sin(ang) * sp2 - 120f;
            sp.life = sp.maxLife = 0.35f + rng.nextFloat() * 0.4f;
            sp.size = 3f + rng.nextFloat() * 5f;
            sp.color = rng.nextBoolean() ? 0xFFFFE29A : 0xFFFF9F68;
            sparks.add(sp);
        }
        addRing(x, y, 0xFFFFE7B8, 74f);
        if (ko) {
            totalKos++;
            addPopup(f.centerX(), f.centerY() - 90f, "НОКАУТ", 0xFFFF7A6E, 34f);
            for (int s = 0; s < 26; s++) {
                Spark sp = new Spark();
                sp.x = f.centerX();
                sp.y = f.centerY();
                float ang = rng.nextFloat() * 6.283f;
                float sp2 = 120f + rng.nextFloat() * 430f;
                sp.vx = (float) Math.cos(ang) * sp2;
                sp.vy = (float) Math.sin(ang) * sp2 - 100f;
                sp.life = sp.maxLife = 0.5f + rng.nextFloat() * 0.6f;
                sp.size = 3f + rng.nextFloat() * 6f;
                sp.color = 0xFFFFC46B;
                sparks.add(sp);
            }
            shake = Math.max(shake, 16f);
            if (listener != null) listener.onKO();
        }
    }

    private void checkResult() {
        boolean playerAlive = false, enemyAlive = false;
        for (Fighter f : fighters) {
            boolean gone = f.out || f.eliminated;
            if (!gone) {
                if (f.team == 0) playerAlive = true;
                else enemyAlive = true;
            }
        }
        if (!enemyAlive) finish(1, "ПОБЕДА");
        else if (!playerAlive) finish(2, "ПОРАЖЕНИЕ");
    }

    private void finish(int res, String reason) {
        if (result != 0) return;
        result = res;
        resultReason = reason;
        endDelay = 0f;
        shake = Math.max(shake, 12f);
    }

    public void requestDash(float tapX, float tapY) {
        if (countdown > 0f || result != 0) return;
        boolean any = false;
        int idx = 0;
        for (Fighter f : fighters) {
            if (f.team != 0 || !f.canDash()) continue;
            float spread = (idx - 1) * 46f;
            f.dash(tapX + spread, tapY + (idx % 2) * 14f, 860f);
            addRing(f.centerX(), f.centerY(), 0xFFFFE7B8, 46f);
            any = true;
            idx++;
        }
        if (any && listener != null) listener.onDash();
    }

    public void addRing(float x, float y, int color, float maxR) {
        Ring r = new Ring();
        r.x = x;
        r.y = y;
        r.r = 6f;
        r.maxR = maxR;
        r.life = r.maxLife = 0.36f;
        r.color = color;
        rings.add(r);
    }

    public void addPopup(float x, float y, String text, int color, float size) {
        Popup p = new Popup();
        p.x = x;
        p.y = y;
        p.text = text;
        p.color = color;
        p.size = size;
        p.life = p.maxLife = 0.9f;
        popups.add(p);
    }

    private void updateEffects(float dt) {
        for (int i = sparks.size() - 1; i >= 0; i--) {
            Spark s = sparks.get(i);
            s.life -= dt;
            if (s.life <= 0f) {
                sparks.remove(i);
                continue;
            }
            s.vy += 900f * dt;
            s.x += s.vx * dt;
            s.y += s.vy * dt;
            s.vx *= 0.98f;
        }
        for (int i = rings.size() - 1; i >= 0; i--) {
            Ring r = rings.get(i);
            r.life -= dt;
            if (r.life <= 0f) {
                rings.remove(i);
                continue;
            }
            float t = 1f - r.life / r.maxLife;
            r.r = 6f + (r.maxR - 6f) * (1f - (1f - t) * (1f - t));
        }
        for (int i = popups.size() - 1; i >= 0; i--) {
            Popup p = popups.get(i);
            p.life -= dt;
            if (p.life <= 0f) {
                popups.remove(i);
                continue;
            }
            p.y -= 34f * dt;
        }
    }
}
