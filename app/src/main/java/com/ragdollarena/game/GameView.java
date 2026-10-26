package com.ragdollarena.game;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;
import java.util.Random;

/** Game shell: scenes, input, HUD and the prettier parts of the rendering. */
public class GameView extends View implements BattleSim.Listener {

    public static final int MENU = 0, HOWTO = 1, MAP = 2, BATTLE = 3, RESULT = 4,
            UPGRADE = 5, RECRUIT = 6, EVENT = 7, VICTORY = 8, GAMEOVER = 9;

    private float scale = 1f;
    private float W = 1280f, H = 720f;
    private long lastNanos;
    private float time;
    private float stateTime;
    private int state = MENU;

    private RunState run;
    private BattleSim battle;
    private int resultSynced;
    private boolean resultHandled;
    private boolean wasBoss;

    private final Random rnd = new Random();
    private final ArrayList<Fighter> menuCrew = new ArrayList<>();
    private float menuGround;

    private ArrayList<Integer> upgradeChoices = new ArrayList<>();
    private ArrayList<Chars.Data> recruitChoices = new ArrayList<>();
    private Chars.Data pendingRecruit;
    private String eventTitle = "";
    private String eventText = "";

    private final RectF rPrimary = new RectF(), rSecondary = new RectF(), rAction = new RectF();
    private final RectF[] rCards = {new RectF(), new RectF(), new RectF()};
    private final RectF[] rMembers = {new RectF(), new RectF(), new RectF()};
    private final RectF rSkip = new RectF(), rEventOk = new RectF();

    private float flash;
    private float shake;
    private float touchX = -1f, touchY = -1f, touchAge = 9f;

    private final Vibrator vibrator;
    private final SharedPreferences prefs;
    private int bestStage;

    public GameView(Context context) {
        super(context);
        setFocusable(true);
        setKeepScreenOn(true);
        vibrator = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
        prefs = context.getSharedPreferences("ragdoll", Context.MODE_PRIVATE);
        bestStage = prefs.getInt("best", 0);
        lastNanos = System.nanoTime();
    }

    private void go(int s) {
        state = s;
        stateTime = 0f;
    }

    public void pause() {
        lastNanos = System.nanoTime();
    }

    public void resumeGame() {
        lastNanos = System.nanoTime();
    }

    private void makeMenuCrew() {
        menuCrew.clear();
        int start = rnd.nextInt(Chars.ALL.length);
        for (int i = 0; i < 3; i++) {
            Chars.Data d = Chars.ALL[(start + i * 4) % Chars.ALL.length];
            Fighter f = new Fighter(d, 0, W * (0.32f + i * 0.18f), H * 0.9f,
                    1f, 1f, 1f, 1f, 1f);
            f.facing = i % 2 == 0 ? 1 : -1;
            f.bobScale = 1.5f;
            f.showHp = false;
            menuCrew.add(f);
        }
        menuGround = H * 0.9f;
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        scale = Math.min(h / 720f, w / 1200f);
        H = h / scale;
        W = w / scale;
        menuGround = H * 0.9f;
        makeMenuCrew();
    }

    private void buzz(int ms) {
        if (vibrator == null || !vibrator.hasVibrator()) return;
        if (Build.VERSION.SDK_INT >= 26) {
            vibrator.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE));
        } else {
            vibrator.vibrate(ms);
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        long now = System.nanoTime();
        float dt = (now - lastNanos) / 1_000_000_000f;
        lastNanos = now;
        if (dt > 0.05f) dt = 0.05f;
        if (dt < 0f) dt = 0f;
        time += dt;
        stateTime += dt;
        update(dt);

        canvas.save();
        canvas.scale(scale, scale);
        if (state == BATTLE || state == RESULT) {
            drawBattle(canvas);
        } else if (state == MENU || state == HOWTO) {
            drawMenu(canvas);
            if (state == HOWTO) drawHowTo(canvas);
        } else if (state == MAP || state == EVENT) {
            drawMap(canvas);
            if (state == EVENT) drawEvent(canvas);
        } else if (state == UPGRADE) {
            drawUpgrade(canvas);
        } else if (state == RECRUIT) {
            drawRecruit(canvas);
        } else if (state == VICTORY) {
            drawEndScreen(canvas, true);
        } else if (state == GAMEOVER) {
            drawEndScreen(canvas, false);
        }
        drawTouch(canvas);
        canvas.restore();
        invalidate();
    }

    private void update(float dt) {
        if (flash > 0f) flash = Math.max(0f, flash - dt * 3.4f);
        if (shake > 0f) shake = Math.max(0f, shake - dt * 42f);
        if (touchAge < 2f) touchAge += dt;
        if (state == BATTLE && battle != null) {
            battle.update(dt);
            if (battle.result != 0 && !resultHandled) {
                resultHandled = true;
                resultSynced = battle.result;
                run.syncAfterBattle(battle);
                shake = 14f;
            }
            if (resultHandled && battle.endDelay > 1.35f && state == BATTLE) {
                if (resultSynced == 1) {
                    wasBoss = run.current != null && run.current.type == RunState.T_BOSS;
                    upgradeChoices = run.rollUpgrades();
                    go(RESULT);
                } else {
                    recordRun();
                    go(GAMEOVER);
                }
            }
        } else if (state == MENU || state == HOWTO) {
            for (Fighter f : menuCrew) {
                f.walkDir = 0f;
                f.update(dt, menuGround, W * 0.16f, W * 0.84f);
            }
        }
    }

    private void recordRun() {
        if (run == null) return;
        int stage = Math.max(1, Math.min(8, run.row));
        if (stage > bestStage) {
            bestStage = stage;
            prefs.edit().putInt("best", bestStage).apply();
        }
    }

    @Override
    public void onHit(float strength) {
        if (strength > 420f) {
            flash = Math.max(flash, 0.55f);
            buzz(18);
        } else if (strength > 260f) {
            buzz(8);
        }
    }

    @Override
    public void onKO() {
        flash = 0.8f;
        shake = 20f;
        buzz(70);
    }

    @Override
    public void onDash() {
        buzz(6);
    }

    // ---------------------------------------------------------------- MENU

    private void drawMenu(Canvas c) {
        Art.bg(c, W, H, time);
        float cx = W * 0.5f;

        Art.P.reset();
        Art.P.setShader(new android.graphics.RadialGradient(cx, H * 0.25f, H * 0.32f,
                new int[]{0x33FFC857, 0x11FFC857, 0x00000000}, new float[]{0f, 0.5f, 1f},
                android.graphics.Shader.TileMode.CLAMP));
        c.drawCircle(cx, H * 0.25f, H * 0.32f, Art.P);
        Art.P.setShader(null);

        Art.textShadow(c, "РАГДОЛЛ", cx, H * 0.165f, H * 0.125f, Art.CREAM,
                Paint.Align.CENTER, Art.BLACK);
        Art.textShadow(c, "АРЕНА", cx, H * 0.285f, H * 0.12f, Art.ACCENT,
                Paint.Align.CENTER, Art.BLACK);
        Art.text(c, "турнир типичных человечков", cx, H * 0.355f, H * 0.032f,
                0xFFFFE3C0, Paint.Align.CENTER, Art.BOLD_COND);

        rPrimary.set(cx - 195f, H * 0.40f, cx + 195f, H * 0.40f + 66f);
        Art.button(c, rPrimary, "НОВЫЙ ЗАБЕГ", true, time);
        rSecondary.set(cx - 150f, H * 0.40f + 82f, cx + 150f, H * 0.40f + 136f);
        Art.button(c, rSecondary, "КАК ИГРАТЬ", false, time);

        if (bestStage > 0) {
            Art.text(c, "ЛУЧШИЙ ЗАБЕГ: ЭТАП " + bestStage, cx, H * 0.40f + 172f,
                    H * 0.027f, 0xFFD9C6E8, Paint.Align.CENTER, Art.BOLD_COND);
        }

        drawMenuStage(c);

        Art.text(c, "v1.0 · ragdoll physics", 20f, H - 16f, H * 0.022f,
                0x88FFE3C0, Paint.Align.LEFT, Art.REG);
        Art.text(c, "нажми на человечка", W - 20f, H - 16f, H * 0.022f,
                0x88FFE3C0, Paint.Align.RIGHT, Art.REG);
    }

    private void drawMenuStage(Canvas c) {
        float gy = menuGround;
        float l = W * 0.5f - Math.min(330f, W * 0.27f);
        float r = W * 0.5f + Math.min(330f, W * 0.27f);
        Art.P.reset();
        Art.P.setStyle(Paint.Style.FILL);
        Art.P.setShader(new android.graphics.LinearGradient(0, gy, 0, gy + 60f,
                0xFF4A3358, 0xFF22182E, android.graphics.Shader.TileMode.CLAMP));
        Art.RECT.set(l, gy, r, gy + 54f);
        c.drawRoundRect(Art.RECT, 20f, 20f, Art.P);
        Art.P.setShader(null);
        Art.P.setShader(new android.graphics.LinearGradient(l, 0, r, 0,
                new int[]{0xFFF5B561, 0xFFFFE7A8, 0xFFF5B561}, null,
                android.graphics.Shader.TileMode.CLAMP));
        c.drawRect(l, gy - 5f, r, gy + 4f, Art.P);
        Art.P.setShader(null);
        Art.P.setColor(0x66F5B561);
        c.drawRect(l, gy + 40f, r, gy + 44f, Art.P);

        for (Fighter f : menuCrew) {
            Art.fighter(c, f, time);
        }
    }

    private void drawTouch(Canvas c) {
        if (touchAge > 0.5f || touchX < 0f) return;
        float t = touchAge / 0.5f;
        Art.P.reset();
        Art.P.setStyle(Paint.Style.STROKE);
        Art.P.setStrokeWidth(3f * (1f - t) + 1f);
        Art.P.setColor(((int) ((1f - t) * 140) << 24) | 0xFFFFFF);
        c.drawCircle(touchX, touchY, 18f + t * 52f, Art.P);
        Art.P.setStyle(Paint.Style.FILL);
    }

    private void tapFx(float x, float y) {
        touchX = x;
        touchY = y;
        touchAge = 0f;
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if (e.getAction() != MotionEvent.ACTION_DOWN) return true;
        float x = e.getX() / scale;
        float y = e.getY() / scale;
        tapFx(x, y);
        if (state == MENU) {
            if (rPrimary.contains(x, y)) {
                startRun();
            } else if (rSecondary.contains(x, y)) {
                go(HOWTO);
            } else {
                pokeMenu(x, y);
            }
        } else if (state == HOWTO) {
            if (y > H * 0.78f || rSecondary.contains(x, y)) go(MENU);
        } else if (state == MAP) {
            handleMapTouch(x, y);
        } else if (state == BATTLE) {
            battle.requestDash(x, y);
        } else if (state == RESULT) {
            if (rAction.contains(x, y) && resultSynced == 1) {
                if (wasBoss) {
                    run.finishNode();
                    recordRun();
                    go(VICTORY);
                } else {
                    go(UPGRADE);
                }
            }
        } else if (state == UPGRADE) {
            for (int i = 0; i < 3 && i < upgradeChoices.size(); i++) {
                if (rCards[i].contains(x, y)) {
                    run.applyUpgrade(upgradeChoices.get(i));
                    run.finishNode();
                    if (wasBoss) {
                        recordRun();
                        go(VICTORY);
                    } else {
                        go(MAP);
                    }
                    break;
                }
            }
        } else if (state == RECRUIT) {
            handleRecruitTouch(x, y);
        } else if (state == EVENT) {
            if (rEventOk.contains(x, y)) go(MAP);
        } else if (state == VICTORY || state == GAMEOVER) {
            if (rPrimary.contains(x, y)) {
                startRun();
            } else if (rSecondary.contains(x, y)) {
                go(MENU);
            }
        }
        return true;
    }

    private void pokeMenu(float x, float y) {
        Fighter best = null;
        float bd = 1e9f;
        for (Fighter f : menuCrew) {
            float d = Math.abs(f.centerX() - x);
            if (d < 90f && y > H * 0.5f && d < bd) {
                bd = d;
                best = f;
            }
        }
        if (best != null) {
            best.body.applyImpulse(Ragdoll.CHEST, (x - best.centerX()) * 13f, -520f, 1f / 60f);
            best.body.applyImpulse(Ragdoll.PELVIS, (x - best.centerX()) * 7f, -300f, 1f / 60f);
            shake = 4f;
            flash = 0.15f;
            buzz(12);
        }
    }

    private void startRun() {
        run = new RunState(System.nanoTime());
        wasBoss = false;
        resultHandled = false;
        battle = null;
        pendingRecruit = null;
        go(MAP);
    }

    // ---------------------------------------------------------------- MAP

    private float mapTop() {
        return H * 0.135f;
    }

    private float mapBottom() {
        return H - 246f;
    }

    private float nodeX(RunState.Node n) {
        return 100f + n.nx * (W - 200f);
    }

    private float nodeY(RunState.Node n) {
        float top = mapTop();
        float bot = mapBottom();
        return top + (n.row + 0.5f) * ((bot - top) / 8f);
    }

    private void drawMap(Canvas c) {
        Art.bg(c, W, H, time);
        Art.P.reset();
        Art.P.setColor(0x66120E20);
        c.drawRect(0, 0, W, H, Art.P);

        Art.textShadow(c, "ТУРНИР", 42f, 64f, 48f, Art.CREAM, Paint.Align.LEFT, Art.BLACK);
        Art.text(c, "ЭТАП " + Math.min(8, run.row + 1) + " / 8", 44f, 96f, 23f,
                0xFFD9C6E8, Paint.Align.LEFT, Art.BOLD_COND);
        Art.text(c, "ПОБЕД: " + run.battlesWon, W - 42f, 64f, 24f,
                0xFFFFD79A, Paint.Align.RIGHT, Art.BOLD_COND);

        Art.P.reset();
        Art.P.setStyle(Paint.Style.STROKE);
        Art.P.setStrokeWidth(3f);
        for (RunState.Node n : run.nodes) {
            for (RunState.Node m : n.next) {
                boolean open = n.cleared || run.accessible.contains(n);
                Art.P.setColor(open ? 0xAAFFC857 : 0x335F4A70);
                Art.P.setPathEffect(new android.graphics.DashPathEffect(
                        new float[]{9f, 9f}, -time * 22f));
                c.drawLine(nodeX(n), nodeY(n), nodeX(m), nodeY(m), Art.P);
            }
        }
        Art.P.setPathEffect(null);

        for (RunState.Node n : run.nodes) {
            boolean accessible = run.accessible.contains(n);
            boolean selected = run.current != null && run.current == n;
            float x = nodeX(n);
            float y = nodeY(n);
            float r = n.type == RunState.T_BOSS ? 26f : 19f;
            if (n.cleared) {
                Art.P.reset();
                Art.P.setColor(0x55120E20);
                c.drawCircle(x, y, r, Art.P);
                Art.text(c, "✓", x, y + 9f, 24f, 0x99A8FFD0, Paint.Align.CENTER, Art.BOLD);
                continue;
            }
            if (accessible) {
                float pulse = 0.5f + 0.5f * (float) Math.sin(time * 3.4f + n.row);
                Art.P.reset();
                Art.P.setColor(0x22FFC857);
                c.drawCircle(x, y, r + 16f + pulse * 6f, Art.P);
                Art.P.setStyle(Paint.Style.STROKE);
                Art.P.setStrokeWidth(2.5f);
                Art.P.setColor(0xAAFFD97A);
                c.drawCircle(x, y, r + 9f + pulse * 3f, Art.P);
            }
            int fill = n.type == RunState.T_BOSS ? 0xFF5A2C3E
                    : n.type == RunState.T_ELITE ? 0xFF432648
                    : n.type == RunState.T_REST ? 0xFF2B3B4A
                    : n.type == RunState.T_RECRUIT ? 0xFF2E3F3C : 0xFF2C2342;
            Art.P.reset();
            Art.P.setStyle(Paint.Style.FILL);
            Art.P.setColor(selected ? 0xFFFFC857 : 0xFF1A1428);
            c.drawCircle(x, y, r + 4f, Art.P);
            Art.P.setColor(fill);
            c.drawCircle(x, y, r, Art.P);
            int iconColor = n.type == RunState.T_BOSS ? 0xFFFFD97A
                    : n.type == RunState.T_ELITE ? 0xFFFF9F68
                    : n.type == RunState.T_RECRUIT ? 0xFF9CE8C8
                    : n.type == RunState.T_REST ? 0xFF9CD7E8 : 0xFFFFE3C0;
            Art.icon(c, n.type, x, y, r * 1.25f, iconColor);
        }

        drawMapRoster(c);
        if (state != EVENT && run.current != null && !run.current.cleared) {
            drawNodeInfo(c, run.current);
        }
    }

    private void drawNodeInfo(Canvas c, RunState.Node n) {
        float l = 24f;
        float r = W - 24f;
        float top = H - 238f;
        Art.panel(c, l, top, r, top + 72f, 18f, 0xE61B1428, 0x44FFC857, 2f);
        Art.icon(c, n.type, l + 44f, top + 36f, 32f, 0xFFFFD97A);
        Art.text(c, typeName(n.type), l + 80f, top + 30f, 21f, Art.CREAM,
                Paint.Align.LEFT, Art.BOLD_COND);
        Art.text(c, typeDesc(n.type), l + 80f, top + 55f, 16f, 0xFFD9C6E8,
                Paint.Align.LEFT, Art.REG);
        rAction.set(r - 198f, top + 10f, r - 12f, top + 62f);
        Art.button(c, rAction, n.type == RunState.T_REST ? "ОТДОХНУТЬ"
                : n.type == RunState.T_RECRUIT ? "СМОТРЕТЬ" : "В БОЙ", true, time);
    }

    private String typeName(int t) {
        switch (t) {
            case RunState.T_ELITE: return "ЭЛИТА";
            case RunState.T_RECRUIT: return "НАБОР В КОМАНДУ";
            case RunState.T_REST: return "ПРИВАЛ";
            case RunState.T_BOSS: return "ФИНАЛ ТУРНИРА";
            default: return "СОРЕВНОВАНИЕ";
        }
    }

    private String typeDesc(int t) {
        switch (t) {
            case RunState.T_ELITE: return "Сильные соперники. Опаснее, но и награда выше.";
            case RunState.T_RECRUIT: return "Выбери нового человечка в свою команду.";
            case RunState.T_REST: return "Восстанавливает 45% здоровья команды.";
            case RunState.T_BOSS: return "Финальная битва. Победи и стань легендой арены.";
            default: return "Команда соперников. Победи, чтобы получить улучшение.";
        }
    }
    private void drawMapRoster(Canvas c) {
        float l = 24f, r = W - 24f, top = H - 152f, bot = H - 16f;
        Art.panel(c, l, top, r, bot, 20f, 0xD91B1428, 0x33FFE3C0, 2f);
        Art.text(c, "ТВОЯ КОМАНДА", l + 22f, top + 24f, 17f, 0xFFD9C6E8,
                Paint.Align.LEFT, Art.BOLD_COND);
        for (int i = 0; i < run.team.size(); i++) {
            RunState.Member m = run.team.get(i);
            float cx = l + 55f + i * 196f;
            float cy = top + 66f;
            Art.portrait(c, m.data, cx, cy, 24f, 1f);
            Art.text(c, m.data.name, cx + 35f, cy - 12f, 18f, Art.CREAM,
                    Paint.Align.LEFT, Art.BOLD_COND);
            float frac = run.hpFraction(m);
            Art.panel(c, cx + 35f, cy + 2f, cx + 155f, cy + 12f, 5f, 0xFF362B47, 0, 0f);
            Art.P.reset();
            Art.P.setColor(frac > 0.5f ? 0xFF76D68C : frac > 0.25f ? 0xFFFFC857 : 0xFFF06C5A);
            Art.RECT.set(cx + 37f, cy + 4f, cx + 37f + 116f * frac, cy + 10f);
            c.drawRoundRect(Art.RECT, 4f, 4f, Art.P);
            Art.text(c, Math.round(m.hp) + "/" + Math.round(run.maxHp(m)), cx + 35f, cy + 27f,
                    14f, 0xFFB9A9CE, Paint.Align.LEFT, Art.REG);
        }
        float sx = W - 300f;
        Art.text(c, "УЛУЧШЕНИЯ", sx, top + 24f, 15f, 0xFFD9C6E8, Paint.Align.LEFT, Art.BOLD_COND);
        Art.text(c, "СИЛА +" + Math.round((run.powMult() - 1f) * 100f) + "%   МАССА +"
                        + Math.round((run.massMult() - 1f) * 100f) + "%",
                sx, top + 48f, 15f, Art.ACCENT, Paint.Align.LEFT, Art.BOLD_COND);
        Art.text(c, "HP +" + Math.round((run.hpMult() - 1f) * 100f) + "%   СКОРОСТЬ +"
                        + Math.round((run.speedMult() - 1f) * 100f) + "%",
                sx, top + 70f, 15f, 0xFF9CE8C8, Paint.Align.LEFT, Art.BOLD_COND);
        Art.text(c, run.finished ? "ФИНАЛ" : "ВЫБЕРИ СЛЕДУЮЩИЙ ЭТАП",
                sx, top + 100f, 14f, 0xFF9C8AB0, Paint.Align.LEFT, Art.REG);
    }

    private void handleMapTouch(float x, float y) {
        if (run == null || run.finished) return;
        if (rAction.contains(x, y) && run.current != null) {
            enterNode(run.current);
            return;
        }
        for (RunState.Node n : run.accessible) {
            float dx = x - nodeX(n);
            float dy = y - nodeY(n);
            if (dx * dx + dy * dy < 46f * 46f) {
                if (run.current == n) {
                    enterNode(n);
                } else {
                    run.choose(n);
                    buzz(8);
                }
                return;
            }
        }
    }

    private void enterNode(RunState.Node n) {
        int type = n.type;
        if (type == RunState.T_REST) {
            run.healAll(0.45f);
            run.finishNode();
            eventTitle = "ПРИВАЛ";
            eventText = "Команда перевела дух и подлечилась.\nВсе человечки восстановили 45% здоровья.";
            go(EVENT);
        } else if (type == RunState.T_RECRUIT) {
            recruitChoices = run.recruitCandidates();
            pendingRecruit = null;
            go(RECRUIT);
        } else {
            battle = run.createBattle(W, H, n);
            battle.listener = this;
            resultHandled = false;
            resultSynced = 0;
            wasBoss = type == RunState.T_BOSS;
            go(BATTLE);
        }
    }

    private void drawHowTo(Canvas c) {
        Art.P.reset();
        Art.P.setColor(0xCC0D0A18);
        c.drawRect(0, 0, W, H, Art.P);
        float l = W * 0.5f - 430f;
        float r = W * 0.5f + 430f;
        float top = H * 0.12f;
        float bot = H * 0.88f;
        Art.panel(c, l, top, r, bot, 26f, 0xFF241B3A, 0x55FFC857, 3f);
        Art.text(c, "КАК ИГРАТЬ", W * 0.5f, top + 62f, 44f, Art.CREAM,
                Paint.Align.CENTER, Art.BLACK);
        String[] lines = {
                "1.  Собирай типичных человечков в команду из 3 героев.",
                "2.  На арене тапни в сторону соперника — твой боец сделает рывок.",
                "3.  Рывок сбивает врага с ног и сталкивает его с платформы.",
                "4.  Побеждай в соревнованиях и выбирай улучшения команды.",
                "5.  Каждый забег — новый путь, новый состав и финальный босс.",
                "6.  Арена сжимается: не задерживайся на краю!"
        };
        float y = top + 128f;
        for (String s : lines) {
            Art.text(c, s, l + 46f, y, 23f, 0xFFEADCF2, Paint.Align.LEFT, Art.REG);
            y += 46f;
        }
        Art.text(c, "ПОДСКАЗКА: у каждого героя своя пассивка — читай карточки.",
                W * 0.5f, bot - 104f, 19f, 0xFFFFC857, Paint.Align.CENTER, Art.BOLD_COND);
        rSecondary.set(W * 0.5f - 150f, bot - 80f, W * 0.5f + 150f, bot - 24f);
        Art.button(c, rSecondary, "ПОНЯТНО", true, time);
    }
    // ---------------------------------------------------------------- BATTLE

    private void drawBattle(Canvas c) {
        Art.bg(c, W, H, time);
        c.save();
        if (shake > 0f) {
            c.translate((rnd.nextFloat() - 0.5f) * shake, (rnd.nextFloat() - 0.5f) * shake);
        }
        Art.platform(c, battle, time);

        for (BattleSim.Ring ring : battle.rings) {
            float a = Math.max(0f, ring.life / ring.maxLife);
            Art.P.reset();
            Art.P.setStyle(Paint.Style.STROKE);
            Art.P.setStrokeWidth(4f * a + 1f);
            Art.P.setColor((((int) (a * 170f)) << 24) | (ring.color & 0xFFFFFF));
            c.drawCircle(ring.x, ring.y, ring.r, Art.P);
        }
        Art.P.setStyle(Paint.Style.FILL);

        int n = battle.fighters.size();
        boolean[] done = new boolean[n];
        for (int pass = 0; pass < n; pass++) {
            int best = -1;
            float bestY = -1e9f;
            for (int i = 0; i < n; i++) {
                if (done[i]) continue;
                float py = battle.fighters.get(i).body.y[Ragdoll.PELVIS];
                if (py > bestY) {
                    bestY = py;
                    best = i;
                }
            }
            if (best < 0) break;
            done[best] = true;
            Art.fighter(c, battle.fighters.get(best), time);
        }

        for (BattleSim.Spark s : battle.sparks) {
            float a = Math.max(0f, s.life / s.maxLife);
            Art.P.reset();
            Art.P.setColor((((int) (a * 230f)) << 24) | (s.color & 0xFFFFFF));
            c.drawCircle(s.x, s.y, s.size * (0.5f + a * 0.6f), Art.P);
        }
        for (BattleSim.Popup p : battle.popups) {
            float a = Math.max(0f, Math.min(1f, p.life / p.maxLife * 1.6f));
            Art.textShadow(c, p.text, p.x, p.y, p.size,
                    ((int) (a * 255f) << 24) | (p.color & 0xFFFFFF),
                    Paint.Align.CENTER, Art.BLACK);
        }
        c.restore();

        drawBattleHud(c);
        drawCountdown(c);

        if (flash > 0f) {
            Art.P.reset();
            Art.P.setColor(((int) (flash * 120f) << 24) | 0xFFFFFF);
            c.drawRect(0, 0, W, H, Art.P);
        }
        if (battle.result != 0 && state == BATTLE) {
            float t = Math.min(1f, battle.endDelay / 0.35f);
            String txt = battle.result == 1 ? "ПОБЕДА!" : "ПОРАЖЕНИЕ";
            int col = battle.result == 1 ? 0xFFFFD97A : 0xFFF06C5A;
            Art.P.reset();
            Art.P.setColor(((int) (t * 120f) << 24));
            c.drawRect(0, 0, W, H, Art.P);
            Art.textShadow(c, txt, W * 0.5f, H * 0.43f, 84f + (1f - t) * 50f, col,
                    Paint.Align.CENTER, Art.BLACK);
            Art.text(c, battle.resultReason, W * 0.5f, H * 0.5f, 24f, Art.CREAM,
                    Paint.Align.CENTER, Art.BOLD_COND);
        }
        if (state == RESULT) {
            drawResult(c);
        }
    }

    private void drawCountdown(Canvas c) {
        if (battle.countdown <= 0f) return;
        float n = battle.countdown;
        int number = (int) Math.ceil(n - 0.6f);
        String txt = number <= 0 ? "БОЙ!" : String.valueOf(number);
        float frac = n - 0.6f - (float) Math.floor(n - 0.6f);
        float pulse = 1f + 0.18f * (float) Math.sin(frac * Math.PI);
        Art.textShadow(c, txt, W * 0.5f, H * 0.42f, 96f * pulse,
                number <= 0 ? 0xFFFFD97A : Art.CREAM, Paint.Align.CENTER, Art.BLACK);
        if (number > 0) {
            Art.text(c, "ПРИГОТОВЬСЯ", W * 0.5f, H * 0.49f, 22f, 0xFFD9C6E8,
                    Paint.Align.CENTER, Art.BOLD_COND);
        }
    }

    private void drawBattleHud(Canvas c) {
        float panelR = 210f + run.team.size() * 66f;
        Art.panel(c, 18f, 14f, panelR, 88f, 18f, 0xB31B1428, 0x33FFE3C0, 1.5f);
        Art.text(c, "ТВОЯ КОМАНДА", 34f, 37f, 15f, 0xFF9CE8C8, Paint.Align.LEFT, Art.BOLD_COND);
        for (int i = 0; i < run.team.size(); i++) {
            Fighter f = findPlayerFighter(i);
            float cx = 54f + i * 66f;
            if (f == null) continue;
            Art.portrait(c, f.data, cx, 61f, 18f, f.ko || f.out ? 0.35f : 1f);
            float frac = Math.max(0f, Math.min(1f, f.hp / f.maxHp));
            Art.P.reset();
            Art.P.setColor(0xAA0B0814);
            Art.RECT.set(cx - 24f, 80f, cx + 24f, 86f);
            c.drawRoundRect(Art.RECT, 3f, 3f, Art.P);
            Art.P.setColor(frac > 0.5f ? 0xFF76D68C : frac > 0.25f ? 0xFFFFC857 : 0xFFF06C5A);
            Art.RECT.set(cx - 24f, 80f, cx - 24f + 48f * frac, 86f);
            c.drawRoundRect(Art.RECT, 3f, 3f, Art.P);
        }

        int enemyCount = 0;
        for (Fighter f : battle.fighters) if (f.team == 1) enemyCount++;
        float rpW = 54f + Math.max(1, enemyCount) * 62f;
        float rpL = W - 18f - rpW;
        Art.panel(c, rpL, 14f, W - 18f, 88f, 18f, 0xB31B1428, 0x33FFE3C0, 1.5f);
        Art.text(c, "СОПЕРНИКИ", W - 32f, 37f, 15f, 0xFFFF9F8A, Paint.Align.RIGHT, Art.BOLD_COND);
        int ei = 0;
        for (Fighter f : battle.fighters) {
            if (f.team != 1) continue;
            float cx = W - 58f - ei * 62f;
            Art.portrait(c, f.data, cx, 61f, 17f, f.ko || f.out ? 0.3f : 1f);
            float frac = Math.max(0f, Math.min(1f, f.hp / f.maxHp));
            Art.P.reset();
            Art.P.setColor(0xAA0B0814);
            Art.RECT.set(cx - 21f, 80f, cx + 21f, 86f);
            c.drawRoundRect(Art.RECT, 3f, 3f, Art.P);
            Art.P.setColor(frac > 0.5f ? 0xFF76D68C : frac > 0.25f ? 0xFFFFC857 : 0xFFF06C5A);
            Art.RECT.set(cx - 21f, 80f, cx - 21f + 42f * frac, 86f);
            c.drawRoundRect(Art.RECT, 3f, 3f, Art.P);
            ei++;
        }

        float timer = Math.max(0f, 45f - battle.time);
        String ts = String.format("%d:%02d", (int) timer / 60, (int) timer % 60);
        Art.panel(c, W * 0.5f - 62f, 14f, W * 0.5f + 62f, 76f, 16f, 0xB31B1428, 0x44FFC857, 1.5f);
        Art.text(c, ts, W * 0.5f, 54f, 32f, timer < 10f ? 0xFFF06C5A : Art.CREAM,
                Paint.Align.CENTER, Art.BLACK);
        String status = battle.time > 38f ? "ПОСЛЕДНИЙ ШАНС!"
                : battle.time > 15f ? "АРЕНА СЖИМАЕТСЯ" : "СОРЕВНОВАНИЕ";
        Art.text(c, status, W * 0.5f, 94f, 14f, battle.time > 15f ? 0xFFFFB37A : 0xFF9C8AB0,
                Paint.Align.CENTER, Art.BOLD_COND);
        if (battle.time > 38f) {
            Art.P.reset();
            Art.P.setShader(new android.graphics.RadialGradient(W * 0.5f, H * 0.55f, W * 0.62f,
                    new int[]{0x00000000, 0x00000000, 0x44F06C5A}, new float[]{0f, 0.55f, 1f},
                    android.graphics.Shader.TileMode.CLAMP));
            c.drawRect(0, 0, W, H, Art.P);
            Art.P.setShader(null);
        }

        float hintY = H - 34f;
        Art.panel(c, W * 0.5f - 320f, H - 58f, W * 0.5f + 300f, H - 12f,
                16f, 0x77140E20, 0x22FFE7A8, 1f);
        Art.P.reset();
        Art.P.setColor(0x33FFE7A8);
        c.drawCircle(W * 0.5f - 152f, hintY - 8f, 6f + (float) Math.sin(time * 4f) * 1.5f, Art.P);
        Art.text(c, "ТАПНИ ПО АРЕНЕ — РЫВОК", W * 0.5f - 138f, hintY, 19f,
                0xCCFFE3C0, Paint.Align.LEFT, Art.BOLD_COND);
        float dx = W * 0.5f + 170f;
        for (Fighter f : battle.fighters) {
            if (f.team != 0) continue;
            float ready = f.ko || f.out ? 0f
                    : 1f - Math.max(0f, Math.min(1f, f.dashTimer / Math.max(0.01f, f.dashCd)));
            Art.P.reset();
            Art.P.setStyle(Paint.Style.STROKE);
            Art.P.setStrokeWidth(3f);
            Art.P.setColor(0x44FFFFFF);
            c.drawCircle(dx, hintY - 7f, 9f, Art.P);
            Art.P.setColor(ready >= 0.99f ? 0xFF9CE8C8 : 0xFFFFC857);
            Art.RECT.set(dx - 9f, hintY - 16f, dx + 9f, hintY + 2f);
            c.drawArc(Art.RECT, -90f, 360f * ready, false, Art.P);
            dx += 26f;
        }
        Art.P.setStyle(Paint.Style.FILL);
    }

    private Fighter findPlayerFighter(int teamIndex) {
        int seen = 0;
        for (Fighter f : battle.fighters) {
            if (f.team != 0) continue;
            if (seen == teamIndex) return f;
            seen++;
        }
        return null;
    }

    private void drawResult(Canvas c) {
        Art.P.reset();
        Art.P.setColor(0xBB0A0714);
        c.drawRect(0, 0, W, H, Art.P);
        float pw = 620f;
        float l = W * 0.5f - pw / 2f;
        float r = W * 0.5f + pw / 2f;
        float top = H * 0.2f;
        float bot = H * 0.78f;
        Art.panel(c, l, top, r, bot, 28f, 0xFF241B3A, 0x66FFC857, 3f);
        Art.text(c, "ПОБЕДА!", W * 0.5f, top + 78f, 60f, 0xFFFFD97A,
                Paint.Align.CENTER, Art.BLACK);
        Art.text(c, "Соперники повержены. Команда получает награду.",
                W * 0.5f, top + 118f, 20f, 0xFFEADCF2, Paint.Align.CENTER, Art.REG);
        for (int i = 0; i < run.team.size(); i++) {
            RunState.Member m = run.team.get(i);
            float cx = W * 0.5f - (run.team.size() - 1) * 55f + i * 110f;
            Art.portrait(c, m.data, cx, top + 196f, 40f, 1f);
            Art.text(c, Math.round(m.hp) + " HP", cx, top + 254f, 17f,
                    0xFF9CE8C8, Paint.Align.CENTER, Art.BOLD_COND);
        }
        Art.text(c, "ПОБЕД В ЗАБЕГЕ: " + (run.battlesWon + 1), W * 0.5f, top + 296f, 19f,
                0xFFD9C6E8, Paint.Align.CENTER, Art.BOLD_COND);
        rAction.set(W * 0.5f - 200f, bot - 82f, W * 0.5f + 200f, bot - 24f);
        Art.button(c, rAction, "ЗАБРАТЬ НАГРАДУ", true, time);
    }

    private void drawUpgrade(Canvas c) {
        Art.bg(c, W, H, time);
        Art.P.reset();
        Art.P.setColor(0x990A0714);
        c.drawRect(0, 0, W, H, Art.P);
        Art.textShadow(c, "НАГРАДА ЗА ПОБЕДУ", W * 0.5f, H * 0.16f, 48f, Art.CREAM,
                Paint.Align.CENTER, Art.BLACK);
        Art.text(c, "Выбери одно улучшение для всей команды", W * 0.5f, H * 0.21f,
                22f, 0xFFD9C6E8, Paint.Align.CENTER, Art.REG);

        float cw = Math.min(320f, (W - 200f) / 3f - 24f);
        float ch = 300f;
        float gap = 28f;
        float total = cw * 3 + gap * 2;
        float startX = W * 0.5f - total / 2f;
        float top = H * 0.29f;
        for (int i = 0; i < 3 && i < upgradeChoices.size(); i++) {
            RunState.Upgrade u = RunState.UPGRADES[upgradeChoices.get(i)];
            float l = startX + i * (cw + gap);
            rCards[i].set(l, top, l + cw, top + ch);
            Art.panel(c, l, top, l + cw, top + ch, 22f, 0xEE241B3A, 0x55FFC857, 2.5f);
            Art.P.reset();
            Art.P.setShader(new android.graphics.RadialGradient(l + cw / 2f, top + 88f, 70f,
                    new int[]{0x66FFC857, 0x11FFC857, 0x00000000}, null,
                    android.graphics.Shader.TileMode.CLAMP));
            c.drawCircle(l + cw / 2f, top + 88f, 70f, Art.P);
            Art.P.setShader(null);
            Art.P.setColor(0xFFFFD97A);
            c.drawCircle(l + cw / 2f, top + 80f, 34f, Art.P);
            Art.text(c, "+", l + cw / 2f, top + 92f, 40f, 0xFF3A2310,
                    Paint.Align.CENTER, Art.BLACK);
            Art.text(c, u.title, l + cw / 2f, top + 152f, 24f, Art.CREAM,
                    Paint.Align.CENTER, Art.BOLD_COND);
            wrap(c, u.desc, l + 24f, top + 188f, cw - 48f, 19f, 0xFFCFC0DF,
                    Paint.Align.CENTER, Art.REG, 25f);
            Art.P.reset();
            Art.P.setColor(0x22FFC857);
            Art.RECT.set(l + 30f, top + ch - 54f, l + cw - 30f, top + ch - 18f);
            c.drawRoundRect(Art.RECT, 12f, 12f, Art.P);
            Art.text(c, "ВЫБРАТЬ", l + cw / 2f, top + ch - 30f, 18f, 0xFFFFC857,
                    Paint.Align.CENTER, Art.BOLD_COND);
        }
        Art.text(c, "Улучшения действуют до конца забега", W * 0.5f, H - 42f, 18f,
                0xFF9C8AB0, Paint.Align.CENTER, Art.REG);
    }

    private void drawEvent(Canvas c) {
        Art.P.reset();
        Art.P.setColor(0xCC0A0714);
        c.drawRect(0, 0, W, H, Art.P);
        float l = W * 0.5f - 330f;
        float r = W * 0.5f + 330f;
        float top = H * 0.22f;
        float bot = H * 0.78f;
        Art.panel(c, l, top, r, bot, 26f, 0xFF241B3A, 0x55FFC857, 3f);
        Art.text(c, eventTitle, W * 0.5f, top + 76f, 46f, 0xFFFFD97A,
                Paint.Align.CENTER, Art.BLACK);
        String[] lines = eventText.split("\n");
        float y = top + 132f;
        for (String s : lines) {
            Art.text(c, s, W * 0.5f, y, 21f, 0xFFEADCF2, Paint.Align.CENTER, Art.REG);
            y += 32f;
        }
        if (run != null) {
            for (int i = 0; i < run.team.size(); i++) {
                RunState.Member m = run.team.get(i);
                float cx = W * 0.5f - (run.team.size() - 1) * 62f + i * 124f;
                Art.portrait(c, m.data, cx, top + 232f, 28f, 1f);
                float frac = run.hpFraction(m);
                Art.panel(c, cx - 34f, top + 268f, cx + 34f, top + 278f, 4f, 0xFF362B47, 0, 0f);
                Art.P.reset();
                Art.P.setColor(frac > 0.5f ? 0xFF76D68C : 0xFFFFC857);
                Art.RECT.set(cx - 32f, top + 270f, cx - 32f + 64f * frac, top + 276f);
                c.drawRoundRect(Art.RECT, 3f, 3f, Art.P);
            }
        }
        rEventOk.set(W * 0.5f - 150f, bot - 76f, W * 0.5f + 150f, bot - 20f);
        Art.button(c, rEventOk, "ПРОДОЛЖИТЬ", true, time);
    }

    private void drawEndScreen(Canvas c, boolean win) {
        Art.bg(c, W, H, time);
        Art.P.reset();
        Art.P.setColor(0xCC0A0714);
        c.drawRect(0, 0, W, H, Art.P);
        if (win) {
            for (int i = 0; i < 44; i++) {
                float x = (i * 137.3f + time * (30f + (i % 7) * 8f)) % (W + 60f) - 30f;
                float y = (i * 97.1f + time * (55f + (i % 5) * 14f)) % (H + 80f) - 40f;
                Art.P.reset();
                Art.P.setColor(i % 3 == 0 ? 0xFFFFD97A : i % 3 == 1 ? 0xFFF06C5A : 0xFF9CE8C8);
                c.save();
                c.translate(x, y);
                c.rotate(time * 90f + i * 23f);
                c.drawRect(-5f, -2.5f, 5f, 2.5f, Art.P);
                c.restore();
            }
        }
        float l = W * 0.5f - 380f;
        float r = W * 0.5f + 380f;
        float top = H * 0.13f;
        float bot = H * 0.87f;
        Art.panel(c, l, top, r, bot, 28f, 0xF2241B3A, win ? 0x66FFC857 : 0x66F06C5A, 3f);
        Art.textShadow(c, win ? "ЛЕГЕНДА АРЕНЫ" : "ЗАБЕГ ОКОНЧЕН",
                W * 0.5f, top + 80f, 50f, win ? 0xFFFFD97A : 0xFFFF9F8A,
                Paint.Align.CENTER, Art.BLACK);
        Art.text(c, win ? "Ты собрал лучших человечков и забрал турнир!"
                        : "Твоя команда пала в бою. Рогалик не прощает слабых.",
                W * 0.5f, top + 124f, 21f, 0xFFEADCF2, Paint.Align.CENTER, Art.REG);

        Art.text(c, "ЭТАП " + Math.min(8, run.row + (win ? 1 : 0)) + " / 8", W * 0.5f, top + 184f,
                26f, Art.CREAM, Paint.Align.CENTER, Art.BOLD_COND);
        Art.text(c, "ПОБЕД: " + run.battlesWon, W * 0.5f, top + 218f, 21f,
                0xFF9CE8C8, Paint.Align.CENTER, Art.BOLD_COND);

        for (int i = 0; i < run.team.size(); i++) {
            RunState.Member m = run.team.get(i);
            float cx = W * 0.5f - (run.team.size() - 1) * 82f + i * 164f;
            Art.portrait(c, m.data, cx, top + 300f, 48f, 1f);
            Art.text(c, m.data.name, cx, top + 368f, 17f, Art.CREAM,
                    Paint.Align.CENTER, Art.BOLD_COND);
            Art.text(c, m.data.perkName, cx, top + 390f, 14f, 0xFFFFC857,
                    Paint.Align.CENTER, Art.REG);
        }
        rPrimary.set(W * 0.5f - 390f, bot - 86f, W * 0.5f - 20f, bot - 26f);
        rSecondary.set(W * 0.5f + 20f, bot - 86f, W * 0.5f + 390f, bot - 26f);
        Art.button(c, rPrimary, "НОВЫЙ ЗАБЕГ", true, time);
        Art.button(c, rSecondary, "В МЕНЮ", false, time);
    }

    private void wrap(Canvas c, String text, float x, float y, float maxW, float size,
                      int color, Paint.Align align, android.graphics.Typeface tf, float lh) {
        Art.P.reset();
        Art.P.setAntiAlias(true);
        Art.P.setTypeface(tf);
        Art.P.setTextSize(size);
        String[] words = text.split(" ");
        StringBuilder line = new StringBuilder();
        float cy = y;
        float drawX = align == Paint.Align.CENTER ? x + maxW / 2f
                : align == Paint.Align.RIGHT ? x + maxW : x;
        for (String w : words) {
            String test = line.length() == 0 ? w : line + " " + w;
            if (Art.P.measureText(test) > maxW && line.length() > 0) {
                Art.text(c, line.toString(), drawX, cy, size, color, align, tf);
                line.setLength(0);
                line.append(w);
                cy += lh;
            } else {
                line.setLength(0);
                line.append(test);
            }
        }
        if (line.length() > 0) Art.text(c, line.toString(), drawX, cy, size, color, align, tf);
    }

    // ---------------------------------------------------------------- RECRUIT

    private void drawRecruit(Canvas c) {
        Art.bg(c, W, H, time);
        Art.P.reset();
        Art.P.setColor(0xAA0A0714);
        c.drawRect(0, 0, W, H, Art.P);
        Art.textShadow(c, "НАБОР В КОМАНДУ", W * 0.5f, H * 0.145f, 48f, Art.CREAM,
                Paint.Align.CENTER, Art.BLACK);
        Art.text(c, "Выбери человечка, который присоединится к твоим бойцам",
                W * 0.5f, H * 0.195f, 21f, 0xFFD9C6E8, Paint.Align.CENTER, Art.REG);
        Art.text(c, "В команде " + run.team.size() + " / 3", W - 50f, H * 0.195f, 18f,
                0xFFFFC857, Paint.Align.RIGHT, Art.BOLD_COND);

        float cw = Math.min(300f, (W - 220f) / 3f - 24f);
        float ch = Math.min(430f, H * 0.57f);
        float gap = 26f;
        float total = cw * 3 + gap * 2;
        float startX = W * 0.5f - total / 2f;
        float top = H * 0.235f;
        for (int i = 0; i < recruitChoices.size() && i < 3; i++) {
            Chars.Data d = recruitChoices.get(i);
            float l = startX + i * (cw + gap);
            rCards[i].set(l, top, l + cw, top + ch);
            Art.panel(c, l, top, l + cw, top + ch, 22f, 0xEE241B3A, 0x44FFC857, 2.5f);
            Art.portrait(c, d, l + cw / 2f, top + 66f, 42f, 1f);
            Art.text(c, d.name, l + cw / 2f, top + 126f, 23f, Art.CREAM,
                    Paint.Align.CENTER, Art.BOLD_COND);
            Art.text(c, d.role, l + cw / 2f, top + 150f, 16f, 0xFF9C8AB0,
                    Paint.Align.CENTER, Art.REG);
            Art.text(c, "«" + d.perkName + "»", l + cw / 2f, top + 182f, 18f, 0xFFFFC857,
                    Paint.Align.CENTER, Art.BOLD_COND);
            wrap(c, d.perkDesc, l + 26f, top + 208f, cw - 52f, 15f, 0xFFCFC0DF,
                    Paint.Align.CENTER, Art.REG, 19f);
            float sy = top + 230f;
            statBar(c, l + 28f, sy, cw - 56f, "HP", d.hp / 150f, 0xFF76D68C);
            statBar(c, l + 28f, sy + 30f, cw - 56f, "СИЛА", d.power / 1.35f, 0xFFF06C5A);
            statBar(c, l + 28f, sy + 60f, cw - 56f, "МАССА", d.mass / 1.4f, 0xFFFFC857);
            statBar(c, l + 28f, sy + 90f, cw - 56f, "СКОРОСТЬ", d.speed / 1.35f, 0xFF74D6C8);
            Art.P.reset();
            Art.P.setColor(0x22FFC857);
            Art.RECT.set(l + 24f, top + ch - 46f, l + cw - 24f, top + ch - 14f);
            c.drawRoundRect(Art.RECT, 12f, 12f, Art.P);
            Art.text(c, "ВЗЯТЬ", l + cw / 2f, top + ch - 23f, 17f, Art.ACCENT,
                    Paint.Align.CENTER, Art.BOLD_COND);
        }

        rSkip.set(W * 0.5f - 130f, H - 74f, W * 0.5f + 130f, H - 22f);
        Art.button(c, rSkip, "ПРОПУСТИТЬ", false, time);

        if (pendingRecruit != null) {
            drawReplace(c);
        }
    }

    private void statBar(Canvas c, float x, float y, float w, String label, float frac, int color) {
        Art.text(c, label, x, y + 11f, 15f, 0xFFCFC0DF, Paint.Align.LEFT, Art.BOLD_COND);
        Art.panel(c, x + 82f, y, x + w, y + 12f, 6f, 0xFF362B47, 0, 0f);
        Art.P.reset();
        Art.P.setColor(color);
        frac = Math.max(0.06f, Math.min(1f, frac));
        Art.RECT.set(x + 84f, y + 2f, x + 84f + (w - 86f) * frac, y + 10f);
        c.drawRoundRect(Art.RECT, 4f, 4f, Art.P);
    }


    private void drawReplace(Canvas c) {
        Art.P.reset();
        Art.P.setColor(0xDD0A0714);
        c.drawRect(0, 0, W, H, Art.P);
        float l = W * 0.5f - 380f;
        float r = W * 0.5f + 380f;
        float top = H * 0.2f;
        float bot = H * 0.8f;
        Art.panel(c, l, top, r, bot, 26f, 0xFF241B3A, 0x55FFC857, 3f);
        Art.text(c, "КОГО ЗАМЕНИТЬ?", W * 0.5f, top + 66f, 38f, Art.CREAM,
                Paint.Align.CENTER, Art.BLACK);
        Art.text(c, pendingRecruit.name + " займёт место выбранного бойца",
                W * 0.5f, top + 102f, 19f, 0xFFD9C6E8, Paint.Align.CENTER, Art.REG);
        for (int i = 0; i < run.team.size(); i++) {
            RunState.Member m = run.team.get(i);
            float cw = 210f;
            float cx = W * 0.5f - (run.team.size() - 1) * 115f + i * 230f;
            float cy = top + 160f;
            rMembers[i].set(cx - cw / 2f, cy, cx + cw / 2f, cy + 170f);
            Art.panel(c, rMembers[i].left, rMembers[i].top, rMembers[i].right, rMembers[i].bottom,
                    18f, 0xEE30233F, 0x44FFC857, 2f);
            Art.portrait(c, m.data, cx, cy + 55f, 32f, 1f);
            Art.text(c, m.data.name, cx, cy + 108f, 17f, Art.CREAM,
                    Paint.Align.CENTER, Art.BOLD_COND);
            Art.text(c, Math.round(m.hp) + " HP", cx, cy + 132f, 15f, 0xFF9CE8C8,
                    Paint.Align.CENTER, Art.REG);
            Art.text(c, "ЗАМЕНИТЬ", cx, cy + 156f, 14f, 0xFFFFC857,
                    Paint.Align.CENTER, Art.BOLD_COND);
        }
        Art.text(c, "Новый герой получит полное здоровье", W * 0.5f, bot - 108f, 17f,
                0xFFFFC857, Paint.Align.CENTER, Art.REG);
        rSkip.set(W * 0.5f - 180f, bot - 78f, W * 0.5f + 180f, bot - 22f);
        Art.button(c, rSkip, "ОСТАВИТЬ КОМАНДУ", false, time);
    }

    private void handleRecruitTouch(float x, float y) {
        if (pendingRecruit != null) {
            for (int i = 0; i < run.team.size() && i < 3; i++) {
                if (rMembers[i].contains(x, y)) {
                    Chars.Data d = pendingRecruit;
                    RunState.Member fresh = new RunState.Member(d);
                    run.team.set(i, fresh);
                    fresh.hp = run.maxHp(fresh);
                    pendingRecruit = null;
                    run.finishNode();
                    go(MAP);
                    return;
                }
            }
            if (rSkip.contains(x, y)) {
                pendingRecruit = null;
            }
            return;
        }
        for (int i = 0; i < recruitChoices.size() && i < 3; i++) {
            if (rCards[i].contains(x, y)) {
                Chars.Data d = recruitChoices.get(i);
                if (run.team.size() < 3) {
                    run.recruit(d);
                    run.finishNode();
                    go(MAP);
                } else {
                    pendingRecruit = d;
                    buzz(10);
                }
                return;
            }
        }
        if (rSkip.contains(x, y)) {
            run.finishNode();
            go(MAP);
        }
    }

}
