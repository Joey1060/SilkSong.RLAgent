using UnityEngine;
using System.Text;

using BepInEx;
using System.Reflection;
using InControl;

public static class HealthManagerUtils {
    private static readonly FieldInfo initHpField =
        typeof(HealthManager).GetField("initHp", BindingFlags.NonPublic | BindingFlags.Instance);

    public static int GetInitHp(HealthManager hm) {
        if (hm == null || initHpField == null)
            return -1; // fallback if something goes wrong

        return (int)initHpField.GetValue(hm);
    }
}

[BepInPlugin("com.joey.combatDebugger", "Combat Debugger", "1.0.0")]
public class CombatDebugger : BaseUnityPlugin {

    private static new readonly BepInEx.Logging.ManualLogSource Logger =
        BepInEx.Logging.Logger.CreateLogSource("CombatDebugger");

    void Awake() {
        Logger.LogInfo("Loaded...");
    }
    void Update() {
        Time.timeScale = 2f;
        if (Input.GetKeyDown(KeyCode.Z)) {
            LogInfo();
        }

        handleInput();
    }
    
    private void handleInput() {
        var hero = HeroController.instance;
        if (hero != null) {
            var field = typeof(HeroController).GetField("inputHandler",
                BindingFlags.NonPublic | BindingFlags.Instance);
            var heroInput = (InputHandler)field.GetValue(hero);
            var deltaTime = Time.deltaTime;
            ulong tick = InputManager.CurrentTick + 1;
            if (Input.GetKey(KeyCode.Keypad3)) {
                heroInput.inputActions.Right.CommitWithState(true, tick, deltaTime);
            }
            else {
                heroInput.inputActions.Right.CommitWithState(false, tick, deltaTime);
            }
        }
    }

    private void LogInfo() {
        Logger.LogInfo("update--------------------------");
        // --- Hornet (player) ---
        var hero = HeroController.instance;
        if (hero != null) {
            var heroHM = hero.GetComponent<HealthManager>();
            var sb = new StringBuilder();
            sb.AppendLine("=== Hornet ===");
            if (heroHM != null) {
                sb.AppendLine($"HP: {heroHM.hp}/{HealthManagerUtils.GetInitHp(heroHM)}");
                sb.AppendLine($"Dead: {heroHM.isDead}");
            }
            sb.AppendLine($"Position: {hero.transform.position}");
            sb.AppendLine($"Velocity: {hero.current_velocity}");
            var fields = typeof(HeroControllerStates).GetFields(BindingFlags.Public | BindingFlags.Instance);
            foreach (var field in fields) {
                var value = field.GetValue(hero.cState);
                sb.AppendLine($"{field.Name} = {value}");
            }
            Logger.LogInfo(sb.ToString());
        }

        // --- Enemies ---
        foreach (var hm in HealthManager.EnumerateActiveEnemies()) {
            if (hm == null) continue;
            var sb = new StringBuilder();
            sb.AppendLine($"=== Enemy: {hm.name} ===");
            sb.AppendLine($"HP: {hm.hp}/{HealthManagerUtils.GetInitHp(hm)}");
            sb.AppendLine($"Dead: {hm.isDead}");
            sb.AppendLine($"Position: {hm.transform.position}");

            var field = typeof(HealthManager).GetField("enemySize", BindingFlags.NonPublic | BindingFlags.Instance);
            var size = (HealthManager.EnemySize)field.GetValue(hm);

            sb.AppendLine($"EnemyType: {hm.EnemyType}, Size: {size}");
            sb.AppendLine($"Invincible: {hm.IsInvincible}");

            var fsms = hm.GetComponents<PlayMakerFSM>();
            foreach (var f in fsms) {
                string fsmName = f.FsmName;
                string state = f.Fsm?.ActiveState?.Name ?? "<null>";
                sb.AppendLine($"FSM {fsmName}: {state}");
            }
            Logger.LogInfo(sb.ToString());
        }
    }
}
