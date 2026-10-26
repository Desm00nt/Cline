package com.ragdollarena.game;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RadialGradient;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.Typeface;

/** All procedural art: backgrounds, little people, icons and UI chrome. */
public final class Art {

    public static final int INK = 0xFF171526;
    public static final int CREAM = 0xFFFFF3DF;
    public static final int ACCENT = 0xFFFFC857;
    public static final int CORAL = 0xFFF06C5A;
    public static final int MINT = 0xFF74D6C8;
    public static final int PLUM = 0xFF3B2140;
    public static final int PANEL = 0xFF241B3A;

    public static final Typeface BLACK = Typeface.create("sans-serif-black", Typeface.NORMAL);
    public static final Typeface BOLD = Typeface.create("sans-serif", Typeface.BOLD);
    public static final Typeface BOLD_COND = Typeface.create("sans-serif-condensed", Typeface.BOLD);
    public static final Typeface REG = Typeface.create("sans-serif", Typeface.NORMAL);

    public static final Paint P = new Paint(Paint.ANTI_ALIAS_FLAG);
    public static final Path PATH = new Path();
    public static final RectF RECT = new RectF();

    private Art() {
    }

    public static void bg(Canvas c, float w, float h, float t) {
        P.reset();
        P.setShader(new LinearGradient(0, 0, 0, h, new int[]{
                0xFF1B1739, 0xFF45224B, 0xFF9C3F54, 0xFFE5795B, 0xFFF7B267
        }, new float[]{0f, 0.34f, 0.64f, 0.84f, 1f}, Shader.TileMode.CLAMP));
        c.drawRect(0, 0, w, h, P);
        P.setShader(null);

        cityLayer(c, w, h, t * 0.6f, 0.0f, 0xFF3B2445, 0.80f, 0.62f);
        cityLayer(c, w, h, t * 1.3f, 46f, 0xFF2A1B38, 0.90f, 0.72f);
        cityLayer(c, w, h, t * 2.1f, 96f, 0xFF1C142B, 1.0f, 0.82f);

        float sunX = w * 0.76f;
        float sunY = h * 0.285f;
        float sunR = h * 0.105f;
        P.setStyle(Paint.Style.FILL);
        P.setShader(new RadialGradient(sunX, sunY, sunR * 2.7f,
                new int[]{0x66FFE1A8, 0x22FFD08A, 0x00000000},
                new float[]{0f, 0.4f, 1f}, Shader.TileMode.CLAMP));
        c.drawCircle(sunX, sunY, sunR * 2.7f, P);
        P.setShader(null);
        P.setColor(0xFFFFD79A);
        c.drawCircle(sunX, sunY, sunR, P);
        P.setColor(0x33FFF3D0);
        c.drawCircle(sunX - sunR * 0.2f, sunY - sunR * 0.2f, sunR * 0.62f, P);

        P.setShader(new LinearGradient(0, h * 0.62f, 0, h,
                0x00000000, 0xCC120E20, Shader.TileMode.CLAMP));
        c.drawRect(0, h * 0.62f, w, h, P);
        P.setShader(null);

        for (int i = 0; i < 26; i++) {
            float fx = (i * 97.13f + t * (8f + (i % 5) * 3f)) % (w + 80f) - 40f;
            float fy = (i * 53.7f) % (h * 0.72f) + (float) Math.sin(t * 0.7f + i) * 14f;
            P.setColor(i % 3 == 0 ? 0x33FFDDAA : 0x22FFFFFF);
            c.drawCircle(fx, fy, 1.2f + (i % 4), P);
        }
    }

    private static void cityLayer(Canvas c, float w, float h, float t, float lift,
                                  int color, float alpha, float top) {
        int count = 16;
        float baseY = h * 0.78f + lift;
        P.setColor(color);
        P.setAlpha((int) (alpha * 255));
        float step = w / (count - 1f);
        for (int i = 0; i < count; i++) {
            float bw = step * (0.55f + ((i * 37) % 10) * 0.05f);
            float bh = h * (0.16f + ((i * 53) % 13) * 0.035f);
            float x = i * step - bw * 0.5f + (float) Math.sin(t * 0.05f + i) * 4f;
            float y = h * top + lift - bh;
            c.drawRect(x, y, x + bw, baseY, P);
            if (i % 3 == 0) {
                c.drawRect(x + bw * 0.4f, y - 22f, x + bw * 0.6f, y, P);
            } else if (i % 4 == 1) {
                c.drawRect(x + bw * 0.42f, y - 40f, x + bw * 0.58f, y, P);
            }
            P.setAlpha((int) (0.55f * 255));
            P.setColor(0xFFFFD58A);
            for (int wy = 0; wy < 4; wy++) {
                for (int wx = 0; wx < 2; wx++) {
                    if (((i * 7 + wy * 3 + wx) % 5) == 0) continue;
                    float lx = x + bw * (0.22f + wx * 0.42f);
                    float ly = y + 12f + wy * (bh * 0.19f);
                    c.drawRect(lx, ly, lx + 4f, ly + 7f, P);
                }
            }
            P.setColor(color);
            P.setAlpha((int) (alpha * 255));
        }
        P.setAlpha(255);
    }

    public static void text(Canvas c, String s, float x, float y, float size,
                            int color, Paint.Align align, Typeface tf) {
        P.reset();
        P.setAntiAlias(true);
        P.setTypeface(tf);
        P.setTextSize(size);
        P.setTextAlign(align);
        P.setColor(color);
        c.drawText(s, x, y, P);
    }

    public static void textShadow(Canvas c, String s, float x, float y, float size,
                                  int color, Paint.Align align, Typeface tf) {
        P.reset();
        P.setAntiAlias(true);
        P.setTypeface(tf);
        P.setTextSize(size);
        P.setTextAlign(align);
        P.setColor(0x66000000);
        c.drawText(s, x + 2f, y + 3f, P);
        P.setColor(color);
        c.drawText(s, x, y, P);
    }

    public static void panel(Canvas c, float l, float t, float r, float b, float rad,
                             int fill, int stroke, float strokeW) {
        RECT.set(l, t, r, b);
        P.reset();
        P.setAntiAlias(true);
        P.setStyle(Paint.Style.FILL);
        P.setColor(fill);
        c.drawRoundRect(RECT, rad, rad, P);
        if (strokeW > 0f) {
            P.setStyle(Paint.Style.STROKE);
            P.setStrokeWidth(strokeW);
            P.setColor(stroke);
            c.drawRoundRect(RECT, rad, rad, P);
        }
    }

    public static void button(Canvas c, RectF r, String label, boolean primary, float time) {
        P.reset();
        P.setAntiAlias(true);
        RECT.set(r.left, r.top + 5f, r.right, r.bottom + 7f);
        P.setColor(0x55000000);
        c.drawRoundRect(RECT, 20f, 20f, P);
        P.setShader(new LinearGradient(0, r.top, 0, r.bottom,
                primary ? 0xFFFFD97A : 0xFF3A2E52,
                primary ? 0xFFE89B45 : 0xFF2A203E, Shader.TileMode.CLAMP));
        RECT.set(r.left, r.top, r.right, r.bottom);
        c.drawRoundRect(RECT, 20f, 20f, P);
        P.setShader(null);
        P.setStyle(Paint.Style.STROKE);
        P.setStrokeWidth(primary ? 3f : 2f);
        P.setColor(primary ? 0x55FFF6D8 : 0x33FFFFFF);
        c.drawRoundRect(RECT, 20f, 20f, P);
        text(c, label, r.centerX(), r.centerY() + 9f, primary ? 27f : 24f,
                primary ? 0xFF3A2310 : CREAM, Paint.Align.CENTER, BOLD_COND);
    }

    public static void icon(Canvas c, int type, float cx, float cy, float size, int color) {
        P.reset();
        P.setAntiAlias(true);
        P.setColor(color);
        P.setStyle(Paint.Style.STROKE);
        P.setStrokeWidth(size * 0.13f);
        P.setStrokeCap(Paint.Cap.ROUND);
        if (type == RunState.T_BATTLE) {
            c.drawLine(cx - size * 0.35f, cy - size * 0.35f,
                    cx + size * 0.35f, cy + size * 0.35f, P);
            c.drawLine(cx + size * 0.35f, cy - size * 0.35f,
                    cx - size * 0.35f, cy + size * 0.35f, P);
            P.setStrokeWidth(size * 0.1f);
            c.drawLine(cx - size * 0.42f, cy + size * 0.3f,
                    cx - size * 0.22f, cy + size * 0.46f, P);
            c.drawLine(cx + size * 0.42f, cy + size * 0.3f,
                    cx + size * 0.22f, cy + size * 0.46f, P);
        } else if (type == RunState.T_ELITE) {
            P.setStyle(Paint.Style.FILL);
            RECT.set(cx - size * 0.4f, cy - size * 0.32f, cx + size * 0.4f, cy + size * 0.36f);
            c.drawRoundRect(RECT, size * 0.3f, size * 0.3f, P);
            P.setColor(0xFF22182E);
            c.drawCircle(cx - size * 0.16f, cy - size * 0.02f, size * 0.09f, P);
            c.drawCircle(cx + size * 0.16f, cy - size * 0.02f, size * 0.09f, P);
            c.drawRect(cx - size * 0.1f, cy + size * 0.12f,
                    cx + size * 0.1f, cy + size * 0.18f, P);
        } else if (type == RunState.T_RECRUIT) {
            P.setStyle(Paint.Style.FILL);
            c.drawCircle(cx - size * 0.1f, cy - size * 0.18f, size * 0.2f, P);
            RECT.set(cx - size * 0.34f, cy - size * 0.02f, cx + size * 0.14f, cy + size * 0.42f);
            c.drawRoundRect(RECT, size * 0.16f, size * 0.16f, P);
            P.setStyle(Paint.Style.STROKE);
            P.setStrokeWidth(size * 0.12f);
            c.drawLine(cx + size * 0.25f, cy, cx + size * 0.25f, cy + size * 0.4f, P);
            c.drawLine(cx + size * 0.07f, cy + size * 0.2f, cx + size * 0.43f, cy + size * 0.2f, P);
        } else if (type == RunState.T_REST) {
            P.setColor(Color.argb(220, Color.red(color), Color.green(color), Color.blue(color)));
            RECT.set(cx - size * 0.36f, cy - size * 0.24f, cx + size * 0.22f, cy + size * 0.2f);
            c.drawRoundRect(RECT, size * 0.12f, size * 0.12f, P);
            P.setStyle(Paint.Style.STROKE);
            P.setStrokeWidth(size * 0.12f);
            RECT.set(cx + size * 0.12f, cy - size * 0.16f, cx + size * 0.44f, cy + size * 0.14f);
            c.drawArc(RECT, 90f, 180f, false, P);
            c.drawLine(cx - size * 0.34f, cy + size * 0.32f, cx + size * 0.3f, cy + size * 0.32f, P);
        } else {
            P.setStyle(Paint.Style.FILL);
            PATH.reset();
            PATH.moveTo(cx - size * 0.38f, cy + size * 0.16f);
            PATH.lineTo(cx - size * 0.3f, cy - size * 0.24f);
            PATH.lineTo(cx - size * 0.12f, cy - size * 0.06f);
            PATH.lineTo(cx, cy - size * 0.34f);
            PATH.lineTo(cx + size * 0.12f, cy - size * 0.06f);
            PATH.lineTo(cx + size * 0.3f, cy - size * 0.24f);
            PATH.lineTo(cx + size * 0.38f, cy + size * 0.16f);
            PATH.lineTo(cx + size * 0.24f, cy + size * 0.38f);
            PATH.lineTo(cx - size * 0.24f, cy + size * 0.38f);
            PATH.close();
            c.drawPath(PATH, P);
        }
    }

    public static void platform(Canvas c, BattleSim sim, float time) {
        float top = sim.groundY;
        float l = sim.platL;
        float r = sim.platR;
        float w = r - l;
        P.reset();
        P.setShader(new LinearGradient(0, top, 0, top + 120f,
                0xFF4A3358, 0xFF22182E, Shader.TileMode.CLAMP));
        RECT.set(l - 16f, top, r + 16f, top + 108f);
        c.drawRoundRect(RECT, 22f, 22f, P);
        P.setShader(null);
        P.setColor(0x66000000);
        RECT.set(l - 4f, top + 84f, r + 4f, top + 118f);
        c.drawRoundRect(RECT, 18f, 18f, P);

        P.reset();
        P.setShader(new LinearGradient(l, 0, r, 0, new int[]{0xFFF5B561, 0xFFFFE7A8, 0xFFF5B561},
                new float[]{0f, 0.5f, 1f}, Shader.TileMode.CLAMP));
        c.drawRect(l - 8f, top - 6f, r + 8f, top + 6f, P);
        P.setShader(null);

        P.setStyle(Paint.Style.STROKE);
        P.setStrokeWidth(3f);
        P.setColor(0x99FFE7A8);
        c.drawLine(l - 8f, top - 8f, r + 8f, top - 8f, P);

        P.setStyle(Paint.Style.FILL);
        for (int i = 0; i < 6; i++) {
            float bx = l + (i + 0.5f) * (w / 6f);
            P.setColor(0x33FFFFFF);
            c.drawRect(bx - 18f, top + 40f, bx + 18f, top + 58f, P);
            P.setColor(0x22FFC857);
            c.drawRect(bx - 18f, top + 62f, bx + 18f, top + 72f, P);
        }

        for (int i = 0; i < 3; i++) {
            float sx = l + w * (0.2f + i * 0.3f);
            float sway = (float) Math.sin(time * 0.5f + i * 1.7f) * 60f;
            P.setShader(new LinearGradient(sx, top - 360f, sx + sway, top,
                    new int[]{0x00FFE7B0, 0x18FFE7B0, 0x00FFE7B0}, null, Shader.TileMode.CLAMP));
            PATH.reset();
            PATH.moveTo(sx - 16f, top - 360f);
            PATH.lineTo(sx + 16f, top - 360f);
            PATH.lineTo(sx + sway + 70f, top);
            PATH.lineTo(sx + sway - 70f, top);
            PATH.close();
            c.drawPath(PATH, P);
            P.setShader(null);
        }

        P.setColor(0xFF171023);
        for (int i = 0; i < 26; i++) {
            float cx = (i * 71.7f + 30f) % (sim.w + 40f) - 20f;
            float cy = sim.h - 16f - ((i * 13) % 22);
            c.drawCircle(cx, cy, 11f, P);
            RECT.set(cx - 12f, cy + 6f, cx + 12f, sim.h + 20f);
            c.drawRoundRect(RECT, 8f, 8f, P);
        }
    }

    public static void fighter(Canvas c, Fighter f, float time) {
        float alpha = f.alpha;
        if (alpha <= 0.01f) return;
        Ragdoll b = f.body;
        Chars.Data d = f.data;
        float px = b.x[Ragdoll.PELVIS];
        float py = b.y[Ragdoll.PELVIS];
        float chestX = b.x[Ragdoll.CHEST];
        float chestY = b.y[Ragdoll.CHEST];
        float headX = b.x[Ragdoll.HEAD];
        float headY = b.y[Ragdoll.HEAD];
        float footLX = b.x[Ragdoll.FOOT_L], footLY = b.y[Ragdoll.FOOT_L];
        float footRX = b.x[Ragdoll.FOOT_R], footRY = b.y[Ragdoll.FOOT_R];
        int body = d.body;
        float limb = body == 1 ? 12f : body == 2 ? 16f : body == 3 ? 18f : 14f;
        float shoulder = body == 1 ? 14f : body == 3 ? 20f : 17f;
        float hip = body == 1 ? 11f : body == 3 ? 16f : 13f;

        int a = (int) (alpha * 255);
        P.reset();
        P.setColor((a / 3) << 24);
        c.drawOval(px - 42f, footLY + 6f, px + 42f, footLY + 20f, P);
        P.setStyle(Paint.Style.STROKE);
        P.setStrokeWidth(2f);
        P.setColor(withAlpha(f.team == 0 ? 0xFF74D6C8 : 0xFFE4575B, alpha * 0.7f));
        c.drawOval(px - 26f, footLY + 7f, px + 26f, footLY + 19f, P);
        P.setStyle(Paint.Style.FILL);

        limb(c, px, py, b.x[Ragdoll.KNEE_L], b.y[Ragdoll.KNEE_L], footLX, footLY,
                limb, 0xFF3A2A38, d.bottom, alpha);
        limb(c, px, py, b.x[Ragdoll.KNEE_R], b.y[Ragdoll.KNEE_R], footRX, footRY,
                limb, 0xFF3A2A38, d.bottom, alpha);
        shoe(c, footLX, footLY, f.facing, d.shoe, alpha);
        shoe(c, footRX, footRY, f.facing, d.shoe, alpha);
        arm(c, chestX - shoulder * 0.8f, chestY + 2f, b.x[Ragdoll.ELBOW_L], b.y[Ragdoll.ELBOW_L],
                b.x[Ragdoll.HAND_L], b.y[Ragdoll.HAND_L], limb * 0.8f, d, alpha, 0.82f);
        torso(c, chestX, chestY, px, py, shoulder, hip, d, alpha, body);
        arm(c, chestX + shoulder * 0.8f, chestY + 2f, b.x[Ragdoll.ELBOW_R], b.y[Ragdoll.ELBOW_R],
                b.x[Ragdoll.HAND_R], b.y[Ragdoll.HAND_R], limb * 0.8f, d, alpha, 1f);

        P.reset();
        P.setColor(withAlpha(d.skin, alpha));
        P.setStrokeCap(Paint.Cap.ROUND);
        P.setStrokeWidth(limb * 0.75f);
        c.drawLine(chestX, chestY - 8f, headX, headY + 6f, P);

        head(c, f, headX, headY, alpha, time);

        if (f.team == 1) {
            P.reset();
            P.setStyle(Paint.Style.STROKE);
            P.setStrokeCap(Paint.Cap.ROUND);
            P.setStrokeWidth(limb * 0.55f);
            P.setColor(withAlpha(0xFFE4575B, alpha * 0.9f));
            c.drawLine((chestX + b.x[Ragdoll.ELBOW_R]) * 0.5f, (chestY + b.y[Ragdoll.ELBOW_R]) * 0.5f,
                    b.x[Ragdoll.ELBOW_R], b.y[Ragdoll.ELBOW_R], P);
        }

        if (f.hitFlash > 0f) {
            P.reset();
            P.setColor(((int) (f.hitFlash * 170f) << 24) | 0xFFFFFF);
            c.drawCircle(headX, headY, 16f, P);
            c.drawOval(chestX - shoulder, chestY - 14f, chestX + shoulder, py + 12f, P);
        }

        if (f.showHp && !f.out && f.hp > 0f) {
            healthBar(c, f, headX, headY - 30f, alpha);
        }
        if (f.ko) {
            text(c, "zZ", headX + 20f * f.facing, headY - 34f, 20f,
                    0xAAFFFFFF, Paint.Align.CENTER, BOLD);
        }
    }

    private static void limb(Canvas c, float x1, float y1, float kx, float ky,
                             float x2, float y2, float width, int outline, int color, float alpha) {
        P.reset();
        P.setStyle(Paint.Style.STROKE);
        P.setStrokeCap(Paint.Cap.ROUND);
        P.setStrokeJoin(Paint.Join.ROUND);
        P.setColor(withAlpha(outline, alpha * 0.55f));
        P.setStrokeWidth(width + 4f);
        PATH.reset();
        PATH.moveTo(x1, y1);
        PATH.lineTo(kx, ky);
        PATH.lineTo(x2, y2);
        c.drawPath(PATH, P);
        P.setColor(withAlpha(color, alpha));
        P.setStrokeWidth(width);
        c.drawPath(PATH, P);
        P.setStyle(Paint.Style.FILL);
        P.setColor(withAlpha(0x33FFFFFF, alpha));
        c.drawCircle(kx, ky, width * 0.34f, P);
    }

    private static void arm(Canvas c, float sx, float sy, float ex, float ey,
                            float hx, float hy, float width, Chars.Data d, float alpha, float shade) {
        P.reset();
        P.setStyle(Paint.Style.STROKE);
        P.setStrokeCap(Paint.Cap.ROUND);
        P.setStrokeJoin(Paint.Join.ROUND);
        P.setColor(withAlpha(0xFF2A1E2E, alpha * 0.5f));
        P.setStrokeWidth(width + 4f);
        c.drawLine(sx, sy, ex, ey, P);
        c.drawLine(ex, ey, hx, hy, P);
        P.setColor(withAlpha(shade(d.top, shade), alpha));
        P.setStrokeWidth(width);
        c.drawLine(sx, sy, ex, ey, P);
        P.setColor(withAlpha(d.skin, alpha));
        P.setStrokeWidth(width * 0.78f);
        c.drawLine(ex, ey, hx, hy, P);
        P.setStyle(Paint.Style.FILL);
        P.setColor(withAlpha(d.skin, alpha));
        c.drawCircle(hx, hy, width * 0.45f, P);
    }

    private static void shoe(Canvas c, float x, float y, int facing, int color, float alpha) {
        P.reset();
        P.setColor(withAlpha(0xFF171220, alpha * 0.55f));
        RECT.set(x - 10f + facing, y - 2f, x + 10f + facing, y + 10f);
        c.drawRoundRect(RECT, 7f, 7f, P);
        P.setColor(withAlpha(color, alpha));
        RECT.set(x - 9f + facing * 3f, y - 5f, x + 9f + facing * 3f, y + 6f);
        c.drawRoundRect(RECT, 6f, 6f, P);
    }

    private static void torso(Canvas c, float cx, float cy, float px, float py,
                              float shoulder, float hip, Chars.Data d, float alpha, int body) {
        float dx = px - cx, dy = py - cy;
        float len = (float) Math.sqrt(dx * dx + dy * dy);
        if (len < 1f) len = 1f;
        float nx = -dy / len, ny = dx / len;
        float sx = cx + nx * shoulder, sy = cy + ny * shoulder;
        float sx2 = cx - nx * shoulder, sy2 = cy - ny * shoulder;
        float hx = px + nx * hip, hy = py + ny * hip;
        float hx2 = px - nx * hip, hy2 = py - ny * hip;
        PATH.reset();
        PATH.moveTo(sx, sy);
        PATH.quadTo(cx + nx * (shoulder + 4f), cy + dy * 0.45f, hx, hy);
        PATH.quadTo(px, py + 8f, hx2, hy2);
        PATH.quadTo(cx - nx * (shoulder + 4f), cy + dy * 0.45f, sx2, sy2);
        PATH.quadTo(cx, cy - 7f, sx, sy);
        PATH.close();

        P.reset();
        P.setStyle(Paint.Style.FILL);
        P.setColor(withAlpha(0xFF171220, alpha * 0.5f));
        c.save();
        c.translate(2.5f, 3f);
        c.drawPath(PATH, P);
        c.restore();
        P.setColor(withAlpha(d.top, alpha));
        c.drawPath(PATH, P);

        P.setColor(withAlpha(shade(d.top, 0.72f), alpha * 0.75f));
        PATH.reset();
        PATH.moveTo(sx2, sy2);
        PATH.quadTo(cx - nx * (shoulder + 4f), cy + dy * 0.45f, hx2, hy2);
        PATH.quadTo(px, py + 8f, px - nx * hip * 0.4f, py + 2f);
        PATH.quadTo(cx - nx * shoulder * 0.3f, cy + 10f, sx2, sy2);
        PATH.close();
        c.drawPath(PATH, P);

        P.setColor(withAlpha(0xFFFFFFFF, alpha * 0.18f));
        PATH.reset();
        PATH.moveTo(cx + nx * shoulder * 0.5f, cy + ny * shoulder * 0.5f);
        PATH.lineTo(cx, cy + 10f);
        PATH.lineTo(cx - nx * shoulder * 0.5f, cy - ny * shoulder * 0.5f);
        PATH.lineTo(cx - nx * shoulder * 0.24f, cy - ny * shoulder * 0.24f);
        PATH.lineTo(cx, cy + 3f);
        PATH.lineTo(cx + nx * shoulder * 0.24f, cy + ny * shoulder * 0.24f);
        PATH.close();
        c.drawPath(PATH, P);

        if (d.accessory == 1 || d.accessory == 6) {
            P.setColor(withAlpha(d.accent, alpha));
            PATH.reset();
            PATH.moveTo(cx, cy - 1f);
            PATH.lineTo(cx + 5f, cy + 6f);
            PATH.lineTo(cx, py - 3f);
            PATH.lineTo(cx - 5f, cy + 6f);
            PATH.close();
            c.drawPath(PATH, P);
        }
        if (d.accessory == 4) {
            P.setColor(withAlpha(d.accent, alpha));
            RECT.set(cx - 11f, cy + 4f, cx + 11f, py + 5f);
            c.drawRoundRect(RECT, 6f, 6f, P);
        }
        if (body >= 2) {
            P.setColor(withAlpha(0xFF000000, alpha * 0.10f));
            c.drawCircle(cx - shoulder * 0.55f, cy + 18f, 9f, P);
        }
    }

    private static void head(Canvas c, Fighter f, float hx, float hy, float alpha, float time) {
        Chars.Data d = f.data;
        int dir = f.facing;
        float r = 16f;
        float fx = hx + dir * 1.6f;
        float fy = hy + 2f;

        if (d.hairStyle == 2) {
            P.setColor(withAlpha(d.hair, alpha));
            RECT.set(hx - r - 3f, hy - 9f, hx + r + 3f, hy + 24f);
            c.drawRoundRect(RECT, 13f, 13f, P);
        }
        P.setColor(withAlpha(shade(d.skin, 0.92f), alpha));
        c.drawCircle(hx - r * 0.92f, hy + 2f, 5.2f, P);
        c.drawCircle(hx + r * 0.92f, hy + 2f, 5.2f, P);

        if (d.hairStyle != 5) {
            P.setColor(withAlpha(d.hair, alpha));
            c.drawCircle(hx - dir * 2f, hy - 3.5f, r + 1.8f, P);
        }
        P.setColor(withAlpha(0x33140F1E, alpha));
        c.drawCircle(fx, fy + 1.5f, r + 1.2f, P);
        P.setColor(withAlpha(d.skin, alpha));
        c.drawCircle(fx, fy, r, P);
        P.setColor(withAlpha(0x33FFFFFF, alpha));
        c.drawCircle(fx - 4f, fy - 5f, r * 0.55f, P);

        P.setColor(withAlpha(0x33F06C5A, alpha));
        c.drawCircle(fx - 8f, fy + 5f, 3.4f, P);
        c.drawCircle(fx + 8f, fy + 5f, 3.4f, P);

        float blink = ((time * 1.4f + f.idleSeed * 3f) % 5.5f) < 0.12f ? 0.15f : 1f;
        P.setColor(withAlpha(0xFF251B22, alpha));
        if (f.ko) {
            P.setStyle(Paint.Style.STROKE);
            P.setStrokeWidth(2.2f);
            c.drawLine(fx - 7f, fy - 4f, fx - 2f, fy + 1f, P);
            c.drawLine(fx - 2f, fy - 4f, fx - 7f, fy + 1f, P);
            c.drawLine(fx + 2f, fy - 4f, fx + 7f, fy + 1f, P);
            c.drawLine(fx + 7f, fy - 4f, fx + 2f, fy + 1f, P);
            P.setStyle(Paint.Style.FILL);
        } else {
            c.drawOval(fx - 8f, fy - 4.4f * blink, fx - 3.2f, fy + 1.6f * blink, P);
            c.drawOval(fx + 3.2f, fy - 4.4f * blink, fx + 8f, fy + 1.6f * blink, P);
            P.setColor(withAlpha(0xCCFFFFFF, alpha));
            c.drawCircle(fx - 6.4f, fy - 3f, 1.1f, P);
            c.drawCircle(fx + 4.8f, fy - 3f, 1.1f, P);
            P.setColor(withAlpha(0xFF251B22, alpha));
            P.setStyle(Paint.Style.STROKE);
            P.setStrokeWidth(1.4f);
            c.drawLine(fx - 7f, fy + 6.4f, fx + 7f, fy + 6.4f, P);
            P.setStyle(Paint.Style.FILL);
        }

        P.setColor(withAlpha(shade(d.hair, 0.75f), alpha));
        RECT.set(fx - 8.4f, fy - 8.4f, fx - 2.4f, fy - 6.8f);
        c.drawRoundRect(RECT, 1.5f, 1.5f, P);
        RECT.set(fx + 2.4f, fy - 8.4f, fx + 8.4f, fy - 6.8f);
        c.drawRoundRect(RECT, 1.5f, 1.5f, P);

        int hs = d.hairStyle;
        P.setColor(withAlpha(d.hair, alpha));
        if (hs == 0 || hs == 1) {
            RECT.set(fx - r - 1f, fy - r - 3f, fx + r + 1f, fy - 2f);
            c.drawArc(RECT, 180f, 180f, true, P);
            PATH.reset();
            PATH.moveTo(fx - r - 1f, fy - 6f);
            PATH.quadTo(fx + dir * 2f, fy - 16f, fx + r * 0.9f, fy - 7f);
            PATH.quadTo(fx + 3f, fy - 9f, fx - r * 0.3f, fy - 3f);
            PATH.close();
            c.drawPath(PATH, P);
        } else if (hs == 2) {
            RECT.set(fx - r - 1.5f, fy - r - 3f, fx + r + 1.5f, fy - 1f);
            c.drawArc(RECT, 175f, 190f, true, P);
            PATH.reset();
            PATH.moveTo(fx - r, fy - 7f);
            PATH.quadTo(fx - 2f, fy - 19f, fx + r, fy - 6f);
            PATH.quadTo(fx + 2f, fy - 10f, fx - r + 2f, fy - 2f);
            PATH.close();
            c.drawPath(PATH, P);
        } else if (hs == 3) {
            RECT.set(fx - r - 1f, fy - r - 3f, fx + r + 1f, fy - 2f);
            c.drawArc(RECT, 180f, 180f, true, P);
            c.drawCircle(hx - dir * 13f, hy - 18f, 8f, P);
            P.setColor(withAlpha(0x33FFFFFF, alpha));
            c.drawCircle(hx - dir * 15f, hy - 20f, 3f, P);
        } else if (hs == 4) {
            c.drawCircle(fx - 9f, fy - 11f, 7f, P);
            c.drawCircle(fx, fy - 15f, 8f, P);
            c.drawCircle(fx + 10f, fy - 10f, 6.5f, P);
        } else if (hs == 5) {
            RECT.set(fx - r, fy - 3f, fx - r + 5f, fy + 6f);
            c.drawRoundRect(RECT, 3f, 3f, P);
            RECT.set(fx + r - 5f, fy - 3f, fx + r, fy + 6f);
            c.drawRoundRect(RECT, 3f, 3f, P);
        }

        headExtra(c, d, hx, hy, fx, fy, r, dir, alpha);
        P.setStyle(Paint.Style.FILL);
    }

    private static void headExtra(Canvas c, Chars.Data d, float hx, float hy,
                                  float fx, float fy, float r, int dir, float alpha) {
        if (d.accessory == 6) {
            P.setColor(withAlpha(d.accent, alpha));
            RECT.set(fx - r - 1.5f, fy - r - 2f, fx + r + 1.5f, fy - 1f);
            c.drawArc(RECT, 180f, 180f, true, P);
            if (dir > 0) {
                RECT.set(fx + 6f, fy - 7f, fx + 30f, fy + 2f);
            } else {
                RECT.set(fx - 30f, fy - 7f, fx - 6f, fy + 2f);
            }
            c.drawRoundRect(RECT, 5f, 5f, P);
            P.setColor(withAlpha(0x33FFFFFF, alpha));
            c.drawCircle(fx, fy - 10f, 2f, P);
        } else if (d.accessory == 2) {
            P.setColor(withAlpha(0xFF2E2A3A, alpha));
            P.setStyle(Paint.Style.STROKE);
            P.setStrokeWidth(4f);
            RECT.set(fx - r - 2f, fy - r - 3f, fx + r + 2f, fy + r * 0.4f);
            c.drawArc(RECT, 185f, 170f, false, P);
            P.setStyle(Paint.Style.FILL);
            RECT.set(fx - r - 6f, fy - 4f, fx - r + 1f, fy + 10f);
            c.drawRoundRect(RECT, 4f, 4f, P);
            RECT.set(fx + r - 1f, fy - 4f, fx + r + 6f, fy + 10f);
            c.drawRoundRect(RECT, 4f, 4f, P);
        } else if (d.accessory == 1) {
            P.setColor(withAlpha(0xCC2A2130, alpha));
            P.setStyle(Paint.Style.STROKE);
            P.setStrokeWidth(2f);
            c.drawCircle(fx - 5.5f, fy - 2f, 5f, P);
            c.drawCircle(fx + 5.5f, fy - 2f, 5f, P);
            c.drawLine(fx - 1f, fy - 2f, fx + 1f, fy - 2f, P);
            P.setStyle(Paint.Style.FILL);
            P.setColor(withAlpha(0x33BDE8FF, alpha));
            c.drawCircle(fx - 5.5f, fy - 2f, 4f, P);
            c.drawCircle(fx + 5.5f, fy - 2f, 4f, P);
        } else if (d.accessory == 3) {
            P.setColor(withAlpha(d.hair, alpha));
            RECT.set(fx - 10f, fy + 1f, fx + 10f, fy + 15f);
            c.drawOval(RECT, P);
            P.setColor(withAlpha(d.skin, alpha));
            RECT.set(fx - 5f, fy + 4f, fx + 5f, fy + 10f);
            c.drawOval(RECT, P);
        }
        if (d.accessory == 5) {
            P.setColor(withAlpha(d.accent, alpha));
            c.drawCircle(hx - r * 0.95f, hy + 8f, 2.6f, P);
            c.drawCircle(hx + r * 0.95f, hy + 8f, 2.6f, P);
        }
    }

    private static void healthBar(Canvas c, Fighter f, float x, float y, float alpha) {
        float w = 50f;
        float frac = Math.max(0f, Math.min(1f, f.hp / Math.max(1f, f.maxHp)));
        P.reset();
        P.setColor(withAlpha(0xAA171220, alpha));
        RECT.set(x - w / 2f - 2f, y - 2f, x + w / 2f + 2f, y + 9f);
        c.drawRoundRect(RECT, 5f, 5f, P);
        int col = frac > 0.55f ? 0xFF76D68C : frac > 0.28f ? 0xFFFFC857 : 0xFFF06C5A;
        P.setColor(withAlpha(col, alpha));
        RECT.set(x - w / 2f, y, x - w / 2f + w * frac, y + 7f);
        c.drawRoundRect(RECT, 4f, 4f, P);
        P.setColor(withAlpha(0x55FFFFFF, alpha));
        c.drawRect(x - w / 2f + 2f, y + 1.4f, x - w / 2f + Math.max(2f, w * frac - 2f), y + 2.6f, P);
        P.setColor(withAlpha(f.team == 0 ? 0xFF74D6C8 : 0xFFE4575B, alpha));
        c.drawCircle(x - w / 2f - 6f, y + 3.5f, 2.4f, P);
    }

    public static void portrait(Canvas c, Chars.Data d, float cx, float cy, float r, float alpha) {
        P.reset();
        P.setAntiAlias(true);
        P.setShader(new LinearGradient(cx, cy - r, cx, cy + r * 1.4f,
                shade(d.top, 1.3f), shade(d.top, 0.5f), Shader.TileMode.CLAMP));
        c.drawCircle(cx, cy, r, P);
        P.setShader(null);
        c.save();
        PATH.reset();
        PATH.addCircle(cx, cy, r, Path.Direction.CW);
        c.clipPath(PATH);

        P.setColor(withAlpha(shade(d.top, 0.85f), alpha));
        RECT.set(cx - r * 0.72f, cy + r * 0.38f, cx + r * 0.72f, cy + r * 1.35f);
        c.drawRoundRect(RECT, r * 0.3f, r * 0.3f, P);
        P.setColor(withAlpha(d.skin, alpha));
        c.drawCircle(cx, cy - r * 0.28f, r * 0.5f, P);
        P.setColor(withAlpha(d.hair, alpha));
        RECT.set(cx - r * 0.52f, cy - r * 0.78f, cx + r * 0.52f, cy - r * 0.24f);
        c.drawArc(RECT, 180f, 180f, true, P);
        if (d.hairStyle == 3) {
            c.drawCircle(cx + r * 0.3f, cy - r * 0.76f, r * 0.2f, P);
        } else if (d.hairStyle == 2) {
            RECT.set(cx - r * 0.56f, cy - r * 0.46f, cx - r * 0.3f, cy + r * 0.32f);
            c.drawRoundRect(RECT, r * 0.12f, r * 0.12f, P);
            RECT.set(cx + r * 0.3f, cy - r * 0.46f, cx + r * 0.56f, cy + r * 0.32f);
            c.drawRoundRect(RECT, r * 0.12f, r * 0.12f, P);
        } else if (d.hairStyle == 4) {
            c.drawCircle(cx - r * 0.3f, cy - r * 0.7f, r * 0.2f, P);
            c.drawCircle(cx + r * 0.12f, cy - r * 0.8f, r * 0.23f, P);
        }
        P.setColor(withAlpha(0xFF251B22, alpha));
        c.drawCircle(cx - r * 0.17f, cy - r * 0.3f, r * 0.055f, P);
        c.drawCircle(cx + r * 0.17f, cy - r * 0.3f, r * 0.055f, P);
        P.setStyle(Paint.Style.STROKE);
        P.setStrokeWidth(r * 0.045f);
        c.drawLine(cx - r * 0.15f, cy - r * 0.08f, cx + r * 0.15f, cy - r * 0.08f, P);
        P.setStyle(Paint.Style.FILL);
        if (d.accessory == 1) {
            P.setStyle(Paint.Style.STROKE);
            P.setStrokeWidth(r * 0.04f);
            c.drawCircle(cx - r * 0.17f, cy - r * 0.3f, r * 0.13f, P);
            c.drawCircle(cx + r * 0.17f, cy - r * 0.3f, r * 0.13f, P);
            c.drawLine(cx - r * 0.04f, cy - r * 0.3f, cx + r * 0.04f, cy - r * 0.3f, P);
            P.setStyle(Paint.Style.FILL);
        } else if (d.accessory == 6) {
            P.setColor(withAlpha(d.accent, alpha));
            RECT.set(cx - r * 0.55f, cy - r * 0.72f, cx + r * 0.55f, cy - r * 0.42f);
            c.drawRoundRect(RECT, r * 0.1f, r * 0.1f, P);
        } else if (d.accessory == 3) {
            P.setColor(withAlpha(d.hair, alpha));
            RECT.set(cx - r * 0.3f, cy - r * 0.1f, cx + r * 0.3f, cy + r * 0.25f);
            c.drawOval(RECT, P);
        }
        c.restore();
        P.setStyle(Paint.Style.STROKE);
        P.setStrokeWidth(2.5f);
        P.setColor(withAlpha(0x88FFF3DF, alpha));
        c.drawCircle(cx, cy, r - 1f, P);
        P.setStyle(Paint.Style.FILL);
    }

    private static int withAlpha(int color, float alpha) {
        int a = (int) (Color.alpha(color) * Math.max(0f, Math.min(1f, alpha)));
        return (a << 24) | (color & 0x00FFFFFF);
    }

    private static int shade(int color, float f) {
        int r = (int) (Color.red(color) * f);
        int g = (int) (Color.green(color) * f);
        int b = (int) (Color.blue(color) * f);
        return Color.rgb(Math.min(255, r), Math.min(255, g), Math.min(255, b));
    }
}

