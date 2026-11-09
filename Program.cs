using System.Text;
using MiniChat.App;
using MiniChat.UI;
using Spectre.Console;
using TorchSharp;
using static TorchSharp.torch;

Console.OutputEncoding = Encoding.UTF8;
Console.InputEncoding = Encoding.UTF8;

// Ctrl+C pendant une barre de progression : Spectre a caché le curseur, on le remet avant de quitter.
Console.CancelKeyPress += (_, _) => AnsiConsole.Cursor.Show();

// Deux dossiers, par défaut à côté de l'exécutable (marche avec "dotnet run", F5 et le dossier publié) :
//   --data <dossier>    texte.txt, dialogues.txt, questions-test.txt
//   --models <dossier>  model/, model-pretrain/, et ce que l'entraînement écrit : checkpoint/, results/
// Un chemin relatif part du dossier courant de la console.
string? dataOption = ReadOption(args, "--data", @"D:\MiniChat\data");
string? modelsOption = ReadOption(args, "--models", @"D:\MiniChat\models");
string dataFolder = Path.GetFullPath(dataOption ?? Path.Combine(AppContext.BaseDirectory, "Data"));
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
// Les poids, les gradients et AdamW restent en float32. TorchSharp 0.105 n'a pas d'autocast (bfloat16).
backends.cuda.matmul.allow_tf32 = true;
backends.cudnn.allow_tf32 = true;
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
static string? StartupCommand(string[] args)
{
    string lr = ReadOption(args, "--lr", "3e-4") is { } rate ? $" --lr {rate}" : "";

    if (args.Contains("--resume")) return "/resume";
    if (ReadOption(args, "--continue", "2000") is { } steps) return $"/continue {steps}{lr}";
    if (ReadOption(args, "--continue-finetune", "200") is { } finetuneSteps) return $"/continue {finetuneSteps} finetune{lr}";
    if (args.Contains("--finetune")) return "/finetune";
    if (args.Contains("--train")) return "/train";
    return null;
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
