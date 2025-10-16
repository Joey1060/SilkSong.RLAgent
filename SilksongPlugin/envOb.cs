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

    private static readonly FieldInfo sceneLoad =
        typeof(GameManager).GetField("sceneLoad", BindingFlags.NonPublic | BindingFlags.Instance);

    public static SceneLoad GetSceneLoad(GameManager gm) {
        return (SceneLoad)sceneLoad.GetValue(gm);
    }

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

    private int frameCount = 0;
    private bool startOb = false;
    private bool isSceneLoaded = false;
    private int prevAction = null;
    private float[] prevState = null;
    private float[] curState = null;

    void Awake() {
        Logger.LogInfo("Loaded...");
    }
    void Update() {
        Time.timeScale = 2f;
        if (Input.GetKeyDown(KeyCode.Z)) {
            LogInfo();
            // Logger.LogInfo()
        }
        // handleInput();
        ListenForTrainingCommand();

        if (startOb && isSceneLoaded) {
            float reward, done;
            // init frame as f0
            // The action will only get applied at the next frame, f1
            // suppose sample every k frames
            // that means the action will be applied after f1, f2 ... fk+1
            // and the effect can be observed at f2, f3 ... fk+1
            // let k be 2 here, so the state will be sampled for every two frames
            // f0: ob s0
            // f1: apply a0
            // f2: ob s1 (to keep key pressing down, also apply a0 here)
            // f3: apply a1
            // f4: ob s2, apply a1....
            if (frameCount % 2 == 0) {
                curState = GetCurState();
                if (prevState != null) {
                    (reward, done) = GetReward();
                    // not sure here or in Python...
                    // LoadSample(prevState, reward, prevAction, curState, done);
                }
                SendStateToAgent(curState);
                prevState = curState;
            }
            UpdateActionFromAgent();
            if (prevAction != null) {
                HandleAction(prevAction);
            }
            ++frameCount;
            if (done) {
                ResetScene();
            }
        }
    }

    private void SendStateToAgent(int[] curState) {

    }
    
    private void UpdateActionFromAgent() {
        
    }

    private (float, float) GetReward() {
        float GetDistanceReward(float curX) {
            return curX / 30;
        }
        float reward = 0;
        if (prevState != null) {
            reward = GetDistanceReward(curState[1]) - GetDistanceReward(prevState[1]);
        }
        float done = 0;
        if (curState[1] >= 30) {
            done = 1;
            reward = 1;
        }
        return (reward, done);
    }
    
    private void ResetScene() {
        GameManager.instance.BeginSceneTransition(new GameManager.SceneLoadInfo {
            PreventCameraFadeOut = true,
            WaitForSceneTransitionCameraFade = false,
            EntryGateName = "left2",
            SceneName = "Mosstown_02c",
            Visualization = GameManager.SceneLoadVisualizations.Default,
            AlwaysUnloadUnusedAssets = true,
            IsFirstLevelForPlayer = false,
        });
        isSceneLoaded = false;
        frameCount = 0;
        prevState = null;
        curState = null;
    }

    private void ListenForTrainingCommand() {
        if (Input.GetKeyDown(KeyCode.Keypad0)) {
            ResetScene();
            startOb = true;
            isSceneLoaded = false;
        }

        if (startOb && !isSceneLoaded) {
            var sceneLoad = HealthManagerUtils.GetSceneLoad(GameManager.instance);
            if (sceneLoad == null) {
                isSceneLoaded = true;
            }
        }

        if (isSceneLoaded) {
            // Loaded
        }
    }

    private void HandleAction(int[] modelInputActions) {
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

    private float[] GetCurState() {
        // Logger.LogInfo("update--------------------------");
        // --- Hornet (player) ---
        var hero = HeroController.instance;
        float hornetHP = 0;
        float hornetPosX = 0;
        float hornetPosY = 0;
        float hornetVelX = 0;
        float hornetVelY = 0;
        if (hero != null) {
            var heroHM = hero.GetComponent<HealthManager>();
            // var sb = new StringBuilder();
            // sb.AppendLine("=== Hornet ===");
            if (heroHM != null) {
                hornetHP = heroHM.hp / HealthManagerUtils.GetInitHp(heroHM);
                // sb.AppendLine($"HP: {heroHM.hp}/{HealthManagerUtils.GetInitHp(heroHM)}");
                // sb.AppendLine($"Dead: {heroHM.isDead}");
            }
            // for now just set a fixed max position...
            float MAX_X = 50;
            float MAX_Y = 10;
            hornetPosX = hero.transform.position[0] / MAX_X;
            hornetPosY = hero.transform.position[1] / MAX_Y;
            hornetVelX = hero.current_velocity[0] / hero.DASH_SPEED;
            hornetVelY = hero.current_velocity[1] / hero.DASH_SPEED;
            // sb.AppendLine($"Position: {hero.transform.position}");
            // sb.AppendLine($"Velocity: {hero.current_velocity}");
            // sb.AppendLine($"Dash Speed: {hero.DASH_SPEED}");
            // var fields = typeof(HeroControllerStates).GetFields(BindingFlags.Public | BindingFlags.Instance);
            // foreach (var field in fields) {
            //     var value = field.GetValue(hero.cState);
            //     sb.AppendLine($"{field.Name} = {value}");
            // }
            // Logger.LogInfo(sb.ToString());
        }
        return [hornetHP, hornetPosX, hornetPosY, hornetVelX, hornetVelY];
        // --- Enemies ---
        // foreach (var hm in HealthManager.EnumerateActiveEnemies()) {
        //     if (hm == null) continue;
        //     var sb = new StringBuilder();
        //     sb.AppendLine($"=== Enemy: {hm.name} ===");
        //     sb.AppendLine($"HP: {hm.hp}/{HealthManagerUtils.GetInitHp(hm)}");
        //     sb.AppendLine($"Dead: {hm.isDead}");
        //     sb.AppendLine($"Position: {hm.transform.position}");

        //     var field = typeof(HealthManager).GetField("enemySize", BindingFlags.NonPublic | BindingFlags.Instance);
        //     var size = (HealthManager.EnemySize)field.GetValue(hm);

        //     sb.AppendLine($"EnemyType: {hm.EnemyType}, Size: {size}");
        //     sb.AppendLine($"Invincible: {hm.IsInvincible}");

        //     var fsms = hm.GetComponents<PlayMakerFSM>();
        //     foreach (var f in fsms) {
        //         string fsmName = f.FsmName;
        //         string state = f.Fsm?.ActiveState?.Name ?? "<null>";
        //         sb.AppendLine($"FSM {fsmName}: {state}");
        //     }
        //     Logger.LogInfo(sb.ToString());
        // }
    }
}
