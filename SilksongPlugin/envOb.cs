using UnityEngine;
using System;
using System.Text;
using System.Reflection;
using System.Collections.Generic;
using BepInEx;
using InControl;
using System.Net;
using System.Net.Sockets;
using HutongGames.PlayMaker.Actions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Threading.Tasks;

public static class HealthManagerUtils {
    public static readonly BepInEx.Logging.ManualLogSource Logger =
        BepInEx.Logging.Logger.CreateLogSource("CombatDebugger");

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

public class RLTransition {
    public float[] PrevState { get; set; }
    public float Reward { get; set; }
    public int Action { get; set; }
    public float[] CurState { get; set; }
    public int Done { get; set; }
}


public class RLCommand {
    public RLTransition transition { get; set; }
    public int action { get; set; }
    /// <summary>
    /// 1: startOB
    /// 2: 
    /// 3: send transition to agent
    /// 4: receive action from agent
    /// </summary>
    public int code { get; set; }
}

public class RLTcpServer {
    private Socket server;
    private Socket client;

    public RLTcpServer(int port) {
        server = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        server.Bind(new IPEndPoint(IPAddress.Any, port));
        server.Listen(1);
        server.Blocking = false; // non-blocking
        Debug.Log($"Server listening on port {port}...");
    }

    private bool IsConnected() {
        if (client == null) {
            return false;
        }
        try {
            if (client.Poll(0, SelectMode.SelectRead)) {
                byte[] buffer = new byte[1];
                if (client.Receive(buffer, SocketFlags.Peek) == 0) {
                    return false;
                }
            }
            return true;
        }
        catch (SocketException e) {
            Debug.LogWarning($"TCP connection error: {e.ErrorCode}|{e.Message}");
            // Debug.LogWarning($"stack: {e.StackTrace}");
            return false;
        }
    }

    private void CleanupClient() {
        if (client != null) {
            try {
                client.Close();
            }
            catch (Exception e) {
                Debug.LogWarning($"TCP connection closing error: {e.Message}");
            }
            client = null;
            Debug.Log("Client disconnected, waiting for new connection...");
        }
    }

    public RLCommand Receive() {
        if (client != null && !IsConnected()) {
            CleanupClient();
        }

        // If no client yet, poll for connection
        if (client == null) {
            if (server.Poll(0, SelectMode.SelectRead)) {
                client = server.Accept();
                client.Blocking = false;
                Debug.Log("Client connected!");
            }
            return null;
        }

        // Already connected → check for incoming data
        if (client.Available > 0) {
            byte[] buffer = new byte[1024];
            int bytesRead = client.Receive(buffer);
            if (bytesRead > 0) {
                var str = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                return JsonConvert.DeserializeObject<RLCommand>(str); ;
            }
        }

        return null;
    }

    public void Respond(RLCommand msg) {
        if (client != null && client.Connected) {
            HealthManagerUtils.Logger.LogInfo("send msg to client");
            string json = JsonConvert.SerializeObject(msg);
            byte[] data = Encoding.UTF8.GetBytes(json + "\n");
            client.Send(data);
        }
    }

    public void Close() {
        CleanupClient();
        server?.Close();
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

public class RLController {
    private int frameCount = 0;
    private bool startOb = false;
    private bool isSceneLoaded = false;
    private int prevAction = -1;
    private float[] prevState = null;
    private float[] curState = null;
    private RLTcpServer server = null;
    private int actionNum = 4;
    public RLController() {
        server = new RLTcpServer(8001);
        // HealthManagerUtils.Logger.LogInfo("server running on ")
    }

    public void Close() {
        server.Close();
    }

    private async void TeleportHero() {
        await Task.Delay(2000);
        await Task.Run(() => {
            var oldPos = HeroController.instance.transform.position;
            HeroController.instance.transform.position = new Vector3(10, oldPos.y, oldPos.z);
            isSceneLoaded = true;
        });
    }

    public void Next() {
        var msg = server.Receive();

        if (msg != null) {
            HealthManagerUtils.Logger.LogInfo(msg.action);
            if (msg.code == 1) {
                startOb = true;
                ResetScene();
            }
        }

        if (startOb && !isSceneLoaded) {
            var sceneLoad = HealthManagerUtils.GetSceneLoad(GameManager.instance);
            // HealthManagerUtils.Logger.LogInfo($"waiting for Scene...  {frameCount}");

            if (sceneLoad == null) {
                TeleportHero();
            }
        }

        if (startOb && isSceneLoaded) {
            // HealthManagerUtils.Logger.LogInfo($"Scene Loaded...   {frameCount}");
            float reward = 0;
            int done = 0;
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

                SendStateToAgent(reward, done);
                prevState = curState;
            }
            UpdateActionFromAgent(msg);
            if (prevAction != -1) {
                HandleAction(prevAction);
            }
            ++frameCount;
            if (done == 1) {
                isSceneLoaded = false;
            }
        }
    }

    private void SendStateToAgent(float reward, int done) {
        var msg = new RLCommand();
        msg.code = 3;
        msg.transition = new RLTransition {
            PrevState = prevState,
            Reward = reward,
            Action = prevAction,
            CurState = curState,
            Done = done
        };
        server.Respond(msg);
    }

    private void UpdateActionFromAgent(RLCommand msg) {
        if (msg != null) {
            if (msg.code == 4) {
                prevAction = msg.action;
            }
        }
    }

    private (float, int) GetReward() {
        if (frameCount > 300) {
            return (-1, 1);
        }
        float GetDistanceReward(float curX) {
            return curX / 30;
        }
        float reward = 0;
        if (prevState != null) {
            reward = GetDistanceReward(curState[1]) - GetDistanceReward(prevState[1]);
        }
        int done = 0;
        if (curState[1] >= 0.6) {
            done = 1;
            reward = 1;
        }
        else if (curState[1] <= 0.1) {
            done = 1;
            reward = -1;
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

    private void HandleAction(int modelInputActions) {
        // TODO: change int[] to an int bitmask
        var hero = HeroController.instance;
        if (hero != null) {
            var field = typeof(HeroController).GetField("inputHandler",
                BindingFlags.NonPublic | BindingFlags.Instance);
            var heroInput = (InputHandler)field.GetValue(hero);
            var deltaTime = Time.deltaTime;
            ulong tick = InputManager.CurrentTick + 1;
            for (int i = 0; i < actionNum; ++i) {
                int curActionBit = 1 << i;
                bool curActionState = ((modelInputActions & curActionBit) != 0);
                InputUtil.GetAction(heroInput.inputActions, curActionBit, curActionState, tick, deltaTime);
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
            hornetPosX = hero.transform.position.x / MAX_X;
            hornetPosY = hero.transform.position.y / MAX_Y;
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
        return new float[] { hornetHP, hornetPosX, hornetPosY, hornetVelX, hornetVelY };
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

[BepInPlugin("com.joey.combatDebugger", "Combat Debugger", "1.0.0")]
public class CombatDebugger : BaseUnityPlugin {


    private RLController rLController = null;

    void Awake() {
        // Logger.LogInfo("Loaded...");
        rLController = new RLController();
    }
    void Update() {
        // Time.timeScale = 2f;
        if (Input.GetKeyDown(KeyCode.Z)) {
            // LogInfo();
            // Logger.LogInfo()
        }
        // handleInput();
        // ListenForTrainingCommand();
        rLController.Next();

    }
    void OnApplicationQuit() {
        rLController.Close();
    }
}
