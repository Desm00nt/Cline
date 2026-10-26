package com.ragdollarena.game;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Roguelike state: roster, upgrades and the tournament map. */
public class RunState {

    public static final int T_BATTLE = 0, T_ELITE = 1, T_RECRUIT = 2, T_REST = 3, T_BOSS = 4;

    public static class Member {
        public final Chars.Data data;
        public float hp;

        Member(Chars.Data data) {
            this.data = data;
            this.hp = data.hp;
        }
    }

    public static class Node {
        public int row, col, type;
        public float nx;
        public boolean cleared;
        public final ArrayList<Node> next = new ArrayList<>();
        public int seed;
    }

    public static class Upgrade {
        public final int id;
        public final String title;
        public final String desc;

        Upgrade(int id, String title, String desc) {
            this.id = id;
            this.title = title;
            this.desc = desc;
        }
    }

    public static final Upgrade[] UPGRADES = {
            new Upgrade(0, "Крепкие кости", "+15% здоровья команде"),
            new Upgrade(1, "Тяжёлый кулак", "+14% силы команде"),
            new Upgrade(2, "Центр масс", "+14% массы команде"),
            new Upgrade(3, "Быстрые ноги", "+10% скорости команде"),
            new Upgrade(4, "Обед в столовой", "Лечит команду на 35%"),
            new Upgrade(5, "Командный дух", "+6% ко всему и лечение 20%"),
    };

    public final Random rng;
    public final ArrayList<Member> team = new ArrayList<>();
    public final ArrayList<Node> nodes = new ArrayList<>();
    public ArrayList<Node> accessible = new ArrayList<>();
    public int hpLvl, powLvl, massLvl, spdLvl;
    public int row;
    public int battlesWon;
    public Node current;
    public int seed;
    public boolean finished;

    public RunState(long seed) {
        this.seed = (int) seed;
        this.rng = new Random(seed);
        buildStartingTeam();
        generateMap();
    }

    private void buildStartingTeam() {
        int a = rng.nextInt(Chars.ALL.length);
        int b = rng.nextInt(Chars.ALL.length - 1);
        if (b >= a) b++;
        team.add(new Member(Chars.ALL[a]));
        team.add(new Member(Chars.ALL[b]));
    }

    private void generateMap() {
        int rows = 8;
        nodes.clear();
        ArrayList<ArrayList<Node>> grid = new ArrayList<>();
        for (int r = 0; r < rows; r++) {
            int count = r == rows - 1 ? 1 : 2 + (r > 0 && rng.nextInt(100) < 45 ? 1 : 0);
            ArrayList<Node> rowNodes = new ArrayList<>();
            for (int c = 0; c < count; c++) {
                Node n = new Node();
                n.row = r;
                n.col = c;
                float spread = count == 1 ? 0.5f
                        : 0.5f + (c - (count - 1) / 2f) * 0.36f;
                n.nx = clamp(spread + (rng.nextFloat() - 0.5f) * 0.04f, 0.14f, 0.86f);
                n.type = pickType(r, rows);
                n.seed = rng.nextInt(100000);
                rowNodes.add(n);
                nodes.add(n);
            }
            grid.add(rowNodes);
        }
        for (int r = 0; r < rows - 1; r++) {
            ArrayList<Node> cur = grid.get(r);
            ArrayList<Node> nxt = grid.get(r + 1);
            for (Node n : cur) {
                Node nearest = null;
                float best = 999f;
                for (Node m : nxt) {
                    float d = Math.abs(m.nx - n.nx);
                    if (d < best) {
                        best = d;
                        nearest = m;
                    }
                }
                if (nearest != null && !n.next.contains(nearest)) n.next.add(nearest);
                if (nxt.size() > 1 && rng.nextFloat() < 0.55f) {
                    Node second = null;
                    float best2 = 999f;
                    for (Node m : nxt) {
                        if (m == nearest) continue;
                        float d = Math.abs(m.nx - n.nx);
                        if (d < best2) {
                            best2 = d;
                            second = m;
                        }
                    }
                    if (second != null && best2 < 0.30f) n.next.add(second);
                }
            }
            for (Node m : nxt) {
                boolean hasIn = false;
                for (Node n : cur) if (n.next.contains(m)) hasIn = true;
                if (!hasIn) {
                    Node nearest = null;
                    float best = 999f;
                    for (Node n : cur) {
                        float d = Math.abs(m.nx - n.nx);
                        if (d < best) {
                            best = d;
                            nearest = n;
                        }
                    }
                    if (nearest != null) nearest.next.add(m);
                }
            }
        }
        accessible = new ArrayList<>(grid.get(0));
    }

    private int pickType(int row, int rows) {
        if (row == rows - 1) return T_BOSS;
        if (row == 0) return T_BATTLE;
        float v = rng.nextFloat();
        if (v < 0.48f) return T_BATTLE;
        if (v < 0.64f) return T_RECRUIT;
        if (v < 0.82f) return T_REST;
        return T_ELITE;
    }

    private static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    public int spirit;

    public float hpMult() {
        return 1f + 0.15f * hpLvl + 0.06f * spirit;
    }

    public float powMult() {
        float synergy = 0f;
        for (Member m : team) if (m.data.perk == Chars.PERK_SYNERGY) synergy += 0.12f;
        return 1f + 0.14f * powLvl + synergy + 0.06f * spirit;
    }

    public float massMult() {
        return 1f + 0.14f * massLvl + 0.06f * spirit;
    }

    public float speedMult() {
        return 1f + 0.10f * spdLvl + 0.06f * spirit;
    }

    public float maxHp(Member m) {
        return m.data.hp * hpMult();
    }

    public float hpFraction(Member m) {
        return Math.max(0.05f, Math.min(1f, m.hp / Math.max(1f, maxHp(m))));
    }

    public boolean canChoose(Node n) {
        return accessible.contains(n) && !finished;
    }

    public void choose(Node n) {
        current = n;
    }

    public void finishNode() {
        if (current == null) return;
        current.cleared = true;
        if (current.type == T_BATTLE || current.type == T_ELITE || current.type == T_BOSS) {
            battlesWon++;
        }
        ArrayList<Node> nx = new ArrayList<>(current.next);
        row = current.row + 1;
        accessible = nx;
        current = null;
        if (accessible.isEmpty()) finished = true;
    }

    public void healAll(float fraction) {
        for (Member m : team) {
            m.hp = Math.min(maxHp(m), m.hp + maxHp(m) * fraction);
        }
    }

    public void applyUpgrade(int id) {
        if (id == 0) {
            float old = hpMult();
            hpLvl++;
            float ratio = hpMult() / old;
            for (Member m : team) m.hp = Math.min(maxHp(m), m.hp * ratio);
        } else if (id == 1) {
            powLvl++;
        } else if (id == 2) {
            massLvl++;
        } else if (id == 3) {
            spdLvl++;
        } else if (id == 4) {
            healAll(0.35f);
        } else {
            spirit++;
            healAll(0.2f);
        }
    }

    public ArrayList<Integer> rollUpgrades() {
        ArrayList<Integer> pool = new ArrayList<>();
        for (int i = 0; i < UPGRADES.length; i++) pool.add(i);
        ArrayList<Integer> out = new ArrayList<>();
        for (int i = 0; i < 3 && !pool.isEmpty(); i++) {
            int idx = rng.nextInt(pool.size());
            out.add(pool.remove(idx));
        }
        return out;
    }

    public ArrayList<Chars.Data> recruitCandidates() {
        ArrayList<Chars.Data> pool = new ArrayList<>();
        for (Chars.Data d : Chars.ALL) {
            boolean inTeam = false;
            for (Member m : team) if (m.data.id.equals(d.id)) inTeam = true;
            if (!inTeam) pool.add(d);
        }
        ArrayList<Chars.Data> out = new ArrayList<>();
        if (pool.isEmpty()) {
            for (int i = 0; i < 3; i++) out.add(Chars.ALL[rng.nextInt(Chars.ALL.length)]);
            return out;
        }
        for (int i = 0; i < 3; i++) {
            out.add(pool.remove(rng.nextInt(pool.size())));
            if (pool.isEmpty()) break;
        }
        return out;
    }

    public void recruit(Chars.Data data) {
        Member m = new Member(data);
        m.hp = maxHp(m);
        team.add(m);
    }

    public void syncAfterBattle(BattleSim battle) {
        for (int i = 0; i < team.size() && i < battle.fighters.size(); i++) {
            Fighter f = battle.fighters.get(i);
            if (f.team != 0) continue;
            Member m = team.get(i);
            m.hp = Math.max(maxHp(m) * 0.3f, f.hp);
            m.hp = Math.min(maxHp(m), m.hp);
        }
        healAll(0.12f);
        for (Member m : team) {
            if (m.data.perk == Chars.PERK_REGEN) m.hp = Math.min(maxHp(m), m.hp + 15f);
            m.hp = Math.min(maxHp(m), m.hp);
        }
    }

    public BattleSim createBattle(float w, float h, Node node) {
        float groundY = h * 0.72f;
        float platL = w * 0.5f - Math.min(330f, w * 0.22f);
        float platR = w * 0.5f + Math.min(330f, w * 0.22f);
        ArrayList<Fighter> players = new ArrayList<>();
        for (int i = 0; i < team.size(); i++) {
            Member m = team.get(i);
            Fighter f = new Fighter(m.data, 0, platL + 95f + i * 70f, groundY,
                    hpMult(), powMult(), massMult(), speedMult(), hpFraction(m));
            players.add(f);
        }
        int count = Math.min(3, 1 + (node.row >= 1 ? 1 : 0) + (node.row >= 4 ? 1 : 0));
        if (node.type == T_ELITE || node.type == T_BOSS) count = Math.min(3, count + 1);
        Random er = new Random(seed * 31L + node.seed);
        float hpM = 0.78f + node.row * 0.115f;
        float powM = 0.82f + node.row * 0.085f;
        float massM = 1f + node.row * 0.02f;
        float speedM = 1f + node.row * 0.015f;
        if (node.type == T_ELITE) {
            hpM += 0.16f;
            powM += 0.14f;
        }
        if (node.type == T_BOSS) {
            hpM += 0.34f;
            powM += 0.26f;
        }
        ArrayList<Fighter> enemies = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            Chars.Data d = Chars.ALL[er.nextInt(Chars.ALL.length)];
            float x = platR - 95f - i * 70f;
            Fighter f = new Fighter(d, 1, x, groundY, hpM, powM, massM, speedM, 1f);
            enemies.add(f);
        }
        return new BattleSim(w, h, players, enemies);
    }
}
