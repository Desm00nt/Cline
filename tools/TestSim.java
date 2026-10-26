import com.ragdollarena.game.BattleSim;
import com.ragdollarena.game.Fighter;
import com.ragdollarena.game.Ragdoll;
import com.ragdollarena.game.RunState;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Headless smoke test for the physics battle and the roguelike map. */
public class TestSim {

    public static void main(String[] args) {
        int battles = 0;
        int wins = 0;
        int timeouts = 0;
        float maxDuration = 0f;
        Random rng = new Random(42);

        for (int row = 0; row < 8; row++) {
            for (int iter = 0; iter < 12; iter++) {
                RunState run = new RunState(1000 + row * 100 + iter);
                while (run.team.size() < 3) {
                    run.recruit(run.recruitCandidates().get(0));
                }
                RunState.Node node = new RunState.Node();
                node.row = row;
                node.seed = rng.nextInt(100000);
                node.type = row == 7 ? RunState.T_BOSS : (row % 3 == 2 ? RunState.T_ELITE : RunState.T_BATTLE);
                BattleSim sim = run.createBattle(1600, 720, node);
                float t = 0f;
                float tapTimer = 0.7f;
                while (sim.result == 0 && t < 70f) {
                    tapTimer -= 1f / 60f;
                    if (tapTimer <= 0f && sim.fighting()) {
                        tapTimer = 0.9f;
                        List<Fighter> enemies = sim.team(1);
                        if (!enemies.isEmpty()) {
                            Fighter target = enemies.get(rng.nextInt(enemies.size()));
                            sim.requestDash(target.centerX(), target.centerY() + 10f);
                        }
                    }
                    sim.update(1f / 60f);
                    t += 1f / 60f;
                    validate(sim, row, iter, t);
                }
                if (sim.result == 0) {
                    System.out.println("FAIL: battle did not finish row=" + row + " iter=" + iter);
                    System.exit(1);
                }
                battles++;
                if (sim.timeUp) timeouts++;
                if (sim.result == 1) wins++;
                maxDuration = Math.max(maxDuration, t);
                run.syncAfterBattle(sim);
                for (RunState.Member m : run.team) {
                    if (m.hp <= 0f || Float.isNaN(m.hp)) {
                        System.out.println("FAIL: bad member hp " + m.hp);
                        System.exit(1);
                    }
                }
            }
        }

        // map connectivity: every run must be walkable from row 0 to the boss
        for (int s = 0; s < 300; s++) {
            RunState run = new RunState(s);
            for (int step = 0; step < 12 && !run.finished; step++) {
                if (run.accessible.isEmpty()) {
                    System.out.println("FAIL: no accessible nodes, seed=" + s);
                    System.exit(1);
                }
                RunState.Node pick = run.accessible.get(run.rng.nextInt(run.accessible.size()));
                run.choose(pick);
                run.finishNode();
            }
            if (!run.finished) {
                System.out.println("FAIL: run never finished, seed=" + s);
                System.exit(1);
            }
        }

        System.out.println("OK battles=" + battles + " playerWins=" + wins
                + " timeouts=" + timeouts + " maxDuration=" + String.format("%.1f", maxDuration) + "s");
        System.out.println("OK maps=300");
    }

    private static void validate(BattleSim sim, int row, int iter, float t) {
        for (Fighter f : sim.fighters) {
            for (int i = 0; i < Ragdoll.N; i++) {
                if (Float.isNaN(f.body.x[i]) || Float.isNaN(f.body.y[i])
                        || Float.isInfinite(f.body.x[i]) || Float.isInfinite(f.body.y[i])) {
                    System.out.println("FAIL: NaN body row=" + row + " iter=" + iter + " t=" + t);
                    System.exit(1);
                }
            }
            if (Float.isNaN(f.hp)) {
                System.out.println("FAIL: NaN hp row=" + row + " iter=" + iter);
                System.exit(1);
            }
        }
    }
}
