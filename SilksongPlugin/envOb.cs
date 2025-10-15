using UnityEngine;
using System;
using System.Text;
using System.Reflection;
using System.Collections.Generic;
using BepInEx;
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

public enum ModelInputAction {
    Up = 1,
    Down = 2,
    Left = 4,
    Right = 8
}

public delegate void InputActionInvoker(HeroActions ia, bool state, ulong tick, float deltaTime);

public static class InputUtil {
    private static Dictionary<ModelInputAction, InputActionInvoker> _map =
        new Dictionary<ModelInputAction, InputActionInvoker>
        {
            { ModelInputAction.Up,    (ia, state, tick, deltaTime) => ia.Up.CommitWithState(state, tick, deltaTime) },
            { ModelInputAction.Down,  (ia, state, tick, deltaTime) => ia.Down.CommitWithState(state, tick, deltaTime) },
            { ModelInputAction.Left,  (ia, state, tick, deltaTime) => ia.Left.CommitWithState(state, tick, deltaTime) },
            { ModelInputAction.Right, (ia, state, tick, deltaTime) => ia.Right.CommitWithState(state, tick, deltaTime) }
        };

    public static void GetAction(HeroActions heroInput, int actCode, bool state, ulong tick, float deltaTime) {
        if (!Enum.IsDefined(typeof(ModelInputAction), actCode))
            throw new ArgumentOutOfRangeException(nameof(actCode), "Invalid action code");

        var act = (ModelInputAction)actCode;
        _map[act](heroInput, state, tick, deltaTime);
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

        // handleInput();

        if (Input.GetKeyDown(KeyCode.Keypad0)) {
            GameManager.instance.BeginSceneTransition(new GameManager.SceneLoadInfo {
                PreventCameraFadeOut = true,
                WaitForSceneTransitionCameraFade = false,
                EntryGateName = "left2",
                SceneName = "Mosstown_02c",
                Visualization = GameManager.SceneLoadVisualizations.Default,
                AlwaysUnloadUnusedAssets = true,
                IsFirstLevelForPlayer = false,
            });
        }
    }

    private void handleInput(int[] modelInputActions) {
        // TODO: change int[] to an int bitmask
        var hero = HeroController.instance;
        if (hero != null) {
            var field = typeof(HeroController).GetField("inputHandler",
                BindingFlags.NonPublic | BindingFlags.Instance);
            var heroInput = (InputHandler)field.GetValue(hero);
            var deltaTime = Time.deltaTime;
            ulong tick = InputManager.CurrentTick + 1;
            for (int i = 0; i < modelInputActions.Length; ++i) {
                int curActionCode = 1 << i;
                bool curActionState = (modelInputActions[i] == 1);
                InputUtil.GetAction(heroInput.inputActions, curActionCode, curActionState, tick, deltaTime);
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
