using System.Text;
using MiniChat.App;
using MiniChat.Config;
using MiniChat.Training;
using MiniChat.UI;
using Spectre.Console;
using TorchSharp;
using static TorchSharp.torch;

Console.OutputEncoding = Encoding.UTF8;
Console.InputEncoding = Encoding.UTF8;

// Ctrl+C pendant une barre de progression : Spectre a caché le curseur, on le remet avant de quitter.
Console.CancelKeyPress += (_, _) => AnsiConsole.Cursor.Show();

// Deux dossiers :
//   --data <dossier>    texte.txt, dialogues.txt, questions-test.txt, eval.json. Par défaut Data/ à côté de
//                       l'exécutable s'il existe (dossier publié), sinon Data/ du projet : la compilation ne copie
//                       plus les données dans bin/ (voir MiniChat.csproj).
//   --models <dossier>  model/, model-pretrain/, et ce que l'entraînement écrit : checkpoint/, results/, cache/.
//                       Par défaut à côté de l'exécutable.
// Un chemin relatif part du dossier courant de la console.
string? dataOption = ReadOption(args, "--data", @"D:\MiniChat\data");
string? modelsOption = ReadOption(args, "--models", @"D:\MiniChat\models");
string dataFolder = Path.GetFullPath(dataOption ?? DefaultDataFolder());
string modelsFolder = Path.GetFullPath(modelsOption ?? AppContext.BaseDirectory);

if (!Directory.Exists(dataFolder))
{
    Messages.Error($"Dossier de données introuvable : {dataFolder}");
    return;
}

// GPU NVIDIA s'il y en a un d'utilisable, sinon CPU. Tout le reste suit ce device :
// le modèle (.to(device)), les batchs (GetBatch) et les tenseurs créés pendant les calculs.
var device = cuda.is_available() ? CUDA : CPU;

// TF32 : les multiplications de matrices float32 passent par les Tensor Cores (GPU RTX 30xx et plus récents),
// avec une mantisse de 10 bits au lieu de 23. Mesuré : 0,53 -> 0,42 s par étape, loss identique à 0,001 près.
// Les poids, les gradients et AdamW restent en float32. TorchSharp n'a pas d'autocast (bfloat16) : ni la 0.105 utilisée ici,
// ni les 0.106 et 0.107 (vérifié en novembre 2026).
backends.cuda.matmul.allow_tf32 = true;
backends.cudnn.allow_tf32 = true;

// --benchmark [taille] : la vitesse d'entraînement avec des micro-batchs de 4, 8 et 16 (voir SpeedBenchmark), puis on quitte.
// Avant de charger un modèle : la mémoire GPU est toute pour la mesure. Chaque mesure tourne dans un processus à part
// (--benchmark-measure), qui renvoie son résultat sur une ligne commençant par BenchmarkResult.
const string BenchmarkResult = "MESURE ";
if (args.Contains("--benchmark"))
{
    RunBenchmark(args, device);
    return;
}
if (args.Contains("--benchmark-measure"))
{
    RunBenchmarkMeasure(args, device);
    return;
}

var workbench = new Workbench(dataFolder, modelsFolder, device);

// Les réglages (température, top-k, top-p, modèle par défaut) : settings.json, à côté des modèles.
string settingsPath = Path.Combine(modelsFolder, Settings.FileName);
var warnings = new List<string>();
var settings = Settings.Load(settingsPath, warnings);
warnings.ForEach(Messages.Warning);
workbench.DefaultModel = settings.DefaultModel;

// Le modèle par défaut s'il existe. Sinon, pas d'entraînement automatique : le cadre d'accueil propose /train.
Messages.Spin("Chargement du modèle...", () => { workbench.LoadSaved(); return 0; });

new NelShell(workbench, settings, settingsPath).Run(StartupCommand(args));

// ---------------------------------------------------------------------------

// Les anciennes options deviennent la commande de Nel Code qui fait la même chose, lancée au démarrage :
//   --train                    /train
//   --finetune                 /finetune
//   --resume                   /resume
//   --continue <pas>           /continue <pas>            (pré-entraînement en plus, depuis model-pretrain/)
//   --continue-finetune <pas>  /continue <pas> finetune   (fine-tuning en plus, depuis model/)
//   --lr <vitesse>             ... --lr <vitesse>         (vitesse de départ d'une continuation)
//   --epochs <passages>        ... --epochs <passages>    (longueur de --train ou --finetune, en passages sur les données)
//   --size <taille>            /train --size <taille>     (small, medium, large : voir ModelConfig.Sizes)
static string? StartupCommand(string[] args)
{
    string lr = ReadOption(args, "--lr", "3e-4") is { } rate ? $" --lr {rate}" : "";
    string epochs = ReadOption(args, "--epochs", "1.1") is { } passes ? $" --epochs {passes}" : "";
    string size = ReadOption(args, "--size", "medium") is { } name ? $" --size {name}" : "";

    if (args.Contains("--resume")) return "/resume";
    if (ReadOption(args, "--continue", "2000") is { } steps) return $"/continue {steps}{lr}";
    if (ReadOption(args, "--continue-finetune", "200") is { } finetuneSteps) return $"/continue {finetuneSteps} finetune{lr}";
    if (args.Contains("--finetune")) return "/finetune" + epochs;
    if (args.Contains("--train")) return "/train" + size + epochs;
    return null;
}

// --benchmark [taille] (large par défaut) : tokens/s par micro-batch, pour le pré-entraînement et le fine-tuning.
static void RunBenchmark(string[] args, Device device)
{
    int i = Array.IndexOf(args, "--benchmark");
    string size = i + 1 < args.Length && !args[i + 1].StartsWith("--") ? args[i + 1] : "large";
    if (!ModelConfig.Sizes.TryGetValue(size, out var config))
    {
        Messages.Error($"Taille inconnue : « {size} ». Tailles : {string.Join(", ", ModelConfig.Sizes.Keys)}.");
        return;
    }
    if (device.type != DeviceType.CUDA)
        Messages.Warning("Pas de GPU : la mesure se fait sur le CPU (lent, et sans rapport avec la vitesse sur GPU).");
    if (GpuMemory.Read() is var (usedMb, totalMb))
        Messages.Dim($"Mémoire GPU déjà prise avant la mesure (Windows, autres programmes) : {usedMb:N0} / {totalMb:N0} Mo.");

    Messages.Title($"Vitesse d'entraînement : modèle {size}, {TrainingConfig.PretrainingFor(config).EffectiveBatchSize} extraits " +
                   $"de {config.BlockSize} tokens par étape");
    var measures = SpeedBenchmark.Run([4, 8, 16], (phase, microBatch) => MeasureInChild(size, phase, microBatch), Messages.Info);
    foreach (var phase in measures.GroupBy(m => m.Phase))
        if (phase.Where(m => m.TokensPerSecond is not null).MaxBy(m => m.TokensPerSecond) is { } best)
            Messages.Success($"{phase.Key} : le plus rapide est le micro-batch {best.MicroBatch} ({best.TokensPerSecond:N0} tokens/s).");
}

// Une mesure de --benchmark dans un processus neuf (MiniChat.exe --benchmark-measure <taille> <phase> <micro-batch>),
// qui la renvoie sur sa dernière ligne, en JSON après BenchmarkResult.
static SpeedMeasure MeasureInChild(string size, string phase, int microBatch)
{
    var start = new System.Diagnostics.ProcessStartInfo(Environment.ProcessPath!)
    {
        RedirectStandardOutput = true,
        UseShellExecute = false,
    };
    foreach (string arg in new[] { "--benchmark-measure", size, phase, microBatch.ToString() })
        start.ArgumentList.Add(arg);

    using var child = System.Diagnostics.Process.Start(start)!;
    string output = child.StandardOutput.ReadToEnd();
    child.WaitForExit();
    string? line = output.Split('\n').LastOrDefault(l => l.StartsWith(BenchmarkResult));
    return line is not null && System.Text.Json.JsonSerializer.Deserialize<SpeedMeasure>(line[BenchmarkResult.Length..]) is { } measure
        ? measure
        : new SpeedMeasure(phase, microBatch, null, null, $"la mesure s'est arrêtée sans résultat (code {child.ExitCode})");
}

// La mesure demandée par MeasureInChild : une seule, puis son résultat sur la sortie.
static void RunBenchmarkMeasure(string[] args, Device device)
{
    int i = Array.IndexOf(args, "--benchmark-measure");
    var config = ModelConfig.Sizes[args[i + 1]] with { VocabSize = DatasetFactory.TokenizerVocabSize };
    var measure = SpeedBenchmark.Measure(args[i + 2], config, int.Parse(args[i + 3]), device);
    Console.WriteLine(BenchmarkResult + System.Text.Json.JsonSerializer.Serialize(measure));
}

// Data/ à côté de l'exécutable (dossier publié), sinon Data/ du projet (noté à la compilation, voir MiniChat.csproj).
static string DefaultDataFolder()
{
    string besideExe = Path.Combine(AppContext.BaseDirectory, "Data");
    if (Directory.Exists(besideExe))
        return besideExe;
    return typeof(Workbench).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), false)
        .Cast<System.Reflection.AssemblyMetadataAttribute>()
        .FirstOrDefault(a => a.Key == "ProjectDataFolder")?.Value ?? besideExe;
}

// Valeur d'une option "--nom valeur", ou null si l'option n'est pas là. example : montré si la valeur manque.
static string? ReadOption(string[] args, string name, string example)
{
    int i = Array.IndexOf(args, name);
    if (i < 0) return null;
    if (i + 1 >= args.Length || args[i + 1].StartsWith("--"))
    {
        Messages.Error($"{name} attend une valeur, par exemple : {name} {example}");
        Environment.Exit(1);
    }
    return args[i + 1];
}
