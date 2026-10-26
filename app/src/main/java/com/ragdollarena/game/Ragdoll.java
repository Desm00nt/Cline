package com.ragdollarena.game;

/**
 * Small verlet ragdoll: eleven points (head, chest, hips, elbows, hands,
 * knees, feet) connected by distance constraints. An "active" layer pulls the
 * points towards a procedural pose, which is what makes the little people
 * wobble, run and then go completely limp when they are knocked out.
 */
public class Ragdoll {

    public static final int HEAD = 0, CHEST = 1, PELVIS = 2,
            ELBOW_L = 3, HAND_L = 4, ELBOW_R = 5, HAND_R = 6,
            KNEE_L = 7, FOOT_L = 8, KNEE_R = 9, FOOT_R = 10, N = 11;

    public final float[] x = new float[N];
    public final float[] y = new float[N];
    public final float[] ox = new float[N];
    public final float[] oy = new float[N];
    public final float[] tx = new float[N];
    public final float[] ty = new float[N];
    public final float[] spring = new float[N];
    public final float[] radius = new float[N];

    private static final int[] CA;
    private static final int[] CB;
    private static final float[] CL;
    private static final float[] CS;

    static {
        int[][] pairs = {
                {CHEST, HEAD}, {CHEST, PELVIS}, {CHEST, ELBOW_L}, {ELBOW_L, HAND_L},
                {CHEST, ELBOW_R}, {ELBOW_R, HAND_R}, {PELVIS, KNEE_L}, {KNEE_L, FOOT_L},
                {PELVIS, KNEE_R}, {KNEE_R, FOOT_R}, {KNEE_L, KNEE_R}, {FOOT_L, FOOT_R},
                {HEAD, PELVIS}, {CHEST, PELVIS}, {HEAD, CHEST}
        };
        float[] lens = {20f, 38f, 20f, 22f, 20f, 22f, 21f, 24f, 21f, 24f, 18f, 16f, 58f, 38f, 20f};
        float[] stiff = {1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 0.6f, 0.35f, 0.45f, 1f, 1f};
        CA = new int[lens.length];
        CB = new int[lens.length];
        CL = lens;
        CS = stiff;
        for (int i = 0; i < lens.length; i++) {
            CA[i] = pairs[i][0];
            CB[i] = pairs[i][1];
        }
    }

    public Ragdoll(float pelvisX, float pelvisY, int facing) {
        float[] rx = {0, 0, 0, -17, -21, 17, 21, -10, -11, 10, 11};
        float[] ry = {-58, -38, 0, -30, -8, -30, -8, 20, 43, 20, 43};
        float[] rad = {15f, 16f, 13f, 6.5f, 7f, 6.5f, 7f, 7.5f, 8f, 7.5f, 8f};
        float[] spr = {240f, 240f, 240f, 110f, 60f, 110f, 60f, 170f, 210f, 170f, 210f};
        for (int i = 0; i < N; i++) {
            x[i] = ox[i] = tx[i] = pelvisX + rx[i] * facing;
            y[i] = oy[i] = ty[i] = pelvisY + ry[i];
            radius[i] = rad[i];
            spring[i] = spr[i];
        }
    }

    public void setTarget(int i, float px, float py) {
        tx[i] = px;
        ty[i] = py;
    }

    public void applyImpulse(int i, float dvx, float dvy, float dt) {
        ox[i] -= dvx * dt;
        oy[i] -= dvy * dt;
    }

    public float vx(int i, float dt) {
        return (x[i] - ox[i]) / Math.max(dt, 0.0001f);
    }

    public float vy(int i, float dt) {
        return (y[i] - oy[i]) / Math.max(dt, 0.0001f);
    }

    /** Keeps the target pose roughly in sync while the body is being thrown. */
    public void translateTargets(float dx, float dy) {
        for (int i = 0; i < N; i++) {
            tx[i] += dx;
            ty[i] += dy;
        }
    }

    public void step(float dt, float gravity, float stiffness, float groundY,
                     float platL, float platR, float friction) {
        float dt2 = dt * dt;
        float netX = 0f;
        for (int i = 0; i < N; i++) {
            if (i == PELVIS) continue;
            netX += (tx[i] - x[i]) * spring[i] * stiffness * 0.18f;
        }
        for (int i = 0; i < N; i++) {
            float vxi = (x[i] - ox[i]) * 0.994f;
            float vyi = (y[i] - oy[i]) * 0.994f;
            ox[i] = x[i];
            oy[i] = y[i];
            float k = spring[i] * stiffness;
            float ax = (tx[i] - x[i]) * k * 0.18f;
            if (i == PELVIS) ax -= netX;
            float ay = (ty[i] - y[i]) * k + gravity;
            x[i] += vxi + ax * dt2;
            y[i] += vyi + ay * dt2;
        }
        for (int it = 0; it < 7; it++) {
            solveConstraints();
        }
        // floor of the arena: only particles above the platform footprint collide
        for (int i = 0; i < N; i++) {
            float r = radius[i];
            if (x[i] > platL - r * 0.4f && x[i] < platR + r * 0.4f
                    && y[i] + r > groundY - 0.6f && y[i] < groundY + 26f) {
                y[i] = groundY - r;
                if (oy[i] > y[i]) {
                    oy[i] = y[i];
                }
                float nvx = (x[i] - ox[i]) * (1f - friction);
                ox[i] = x[i] - nvx;
            }
            // hard safety rails, the arena itself has open sides
            x[i] = clamp(x[i], -900f, 4000f);
            y[i] = clamp(y[i], -2000f, 3000f);
        }
    }

    private void solveConstraints() {
        for (int c = 0; c < CA.length; c++) {
            int a = CA[c], b = CB[c];
            float dx = x[b] - x[a];
            float dy = y[b] - y[a];
            float dist = (float) Math.sqrt(dx * dx + dy * dy);
            if (dist < 0.0001f) continue;
            float diff = (dist - CL[c]) / dist * 0.5f * CS[c];
            dx *= diff;
            dy *= diff;
            x[a] += dx;
            y[a] += dy;
            x[b] -= dx;
            y[b] -= dy;
        }
    }

    public void relax(float blend) {
        for (int i = 0; i < N; i++) {
            ox[i] = x[i] + (ox[i] - x[i]) * (1f - blend);
            oy[i] = y[i] + (oy[i] - y[i]) * (1f - blend);
        }
    }

    private static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
}
