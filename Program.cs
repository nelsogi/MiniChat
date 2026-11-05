using System.Text;
using MiniChat.Chat;
using MiniChat.Config;
using MiniChat.Data;
using MiniChat.Evaluation;
using MiniChat.Generation;
using MiniChat.Model;
using MiniChat.Storage;
using MiniChat.Tokenization;
using MiniChat.Training;
using TorchSharp;
using static TorchSharp.torch;

Console.OutputEncoding = Encoding.UTF8;
Console.InputEncoding = Encoding.UTF8;

// Taille du vocabulaire BPE : plus grand = mots entiers plus souvent, mais plus de données nécessaires.
// 4000 : texte.txt fait maintenant des centaines de millions de tokens, de quoi apprendre chaque token.
const int TokenizerVocabSize = 4000;

// Le tokenizer apprend sur un échantillon de texte.txt réparti dans tout le fichier, pas sur le fichier entier :
// ~1 min 30 pour 20 Mo et 4000 tokens, et les fusions apprises ne changent presque plus au-delà.
const int TokenizerSampleMb = 20;

// Deux dossiers, par défaut à côté de l'exécutable (marche avec "dotnet run", F5 et le dossier publié) :
//   --data <dossier>    texte.txt, dialogues.txt, questions-test.txt
//   --models <dossier>  model/, model-pretrain/, et ce que l'entraînement écrit : checkpoint/, results/
// Un chemin relatif part du dossier courant de la console.
string? dataOption = ReadOption(args, "--data", @"D:\MiniChat\data");
string? modelsOption = ReadOption(args, "--models", @"D:\MiniChat\models");

// Continuer un modèle déjà entraîné, avec les données actuelles de data/ (on peut en avoir ajouté) :
//   --continue <pas>           pré-entraînement en plus, depuis model-pretrain/
//   --continue-finetune <pas>  fine-tuning en plus, depuis model/
//   --lr <vitesse>             vitesse de départ (par défaut 3e-4, 1e-4 en fine-tuning : voir TrainingConfig.Continuation)
int? continuePretrainSteps = ReadSteps(args, "--continue");
int? continueFinetuneSteps = ReadSteps(args, "--continue-finetune");
double? continueLearningRate = ReadNumber(args, "--lr", "3e-4");
string dataFolder = Path.GetFullPath(dataOption ?? Path.Combine(AppContext.BaseDirectory, "data"));
string modelsFolder = Path.GetFullPath(modelsOption ?? AppContext.BaseDirectory);

if (!Directory.Exists(dataFolder))
{
    Console.WriteLine($"Dossier de données introuvable : {dataFolder}");
    return;
}
if (dataOption is not null) Console.WriteLine($"Données : {dataFolder}");
if (modelsOption is not null) Console.WriteLine($"Modèles : {modelsFolder}");

string pretrainPath = Path.Combine(dataFolder, "texte.txt");
string dialoguesPath = Path.Combine(dataFolder, "dialogues.txt");
string questionsPath = Path.Combine(dataFolder, "questions-test.txt");

// Sauvegarde régulière de l'entraînement en cours (pour --resume). Supprimée quand l'entraînement finit.
string checkpointFolder = Path.Combine(modelsFolder, "checkpoint");
bool resume = args.Contains("--resume");
if (resume && !Checkpoint.ExistsIn(checkpointFolder))
{
    Console.WriteLine("Rien à reprendre : pas d'entraînement interrompu dans 'checkpoint'.");
    return;
}
CheckpointState? resumeState = resume ? Checkpoint.ReadState(checkpointFolder) : null;

// Deux sauvegardes : le modèle pré-entraîné (base pour refaire le fine-tuning) et le modèle final.
var pretrainStore = new ModelStore(Path.Combine(modelsFolder, "model-pretrain"));
var store = new ModelStore(Path.Combine(modelsFolder, "model"));

// --continue / --continue-finetune : le modèle de départ et son avancement (pas atteint, loss, date).
string? continuePhase = resume ? null
    : continuePretrainSteps is not null ? "pretrain"
    : continueFinetuneSteps is not null ? "finetune"
    : null;
ModelStore? continueStore = continuePhase switch { "pretrain" => pretrainStore, "finetune" => store, _ => null };
if (continueStore is not null && !continueStore.Exists)
{
    Console.WriteLine($"Rien à continuer : pas de modèle dans '{StoreName(continuePhase!)}'. Lance d'abord avec --train.");
    return;
}
TrainingInfo? continueInfo = continueStore?.ReadInfo();

// Chaque entraînement laisse deux fichiers avec la même date dans results/ :
//   <date>.txt       les réponses aux questions de test
//   <date>-loss.csv  les loss mesurées pendant l'entraînement
// Une reprise garde la date de l'entraînement d'origine : le CSV continue au même endroit.
// Une continuation aussi (date gardée dans training.json), mais ses réponses ont leur propre fichier.
string now = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
string runId = resumeState?.RunId ?? continueInfo?.RunId ?? now;
string resultsFolder = Path.Combine(modelsFolder, "results");
var log = new TrainingLog(Path.Combine(resultsFolder, $"{runId}-loss.csv"), questionsPath);
var checkpoint = new Checkpoint(checkpointFolder, runId);
ContinueState? resumedContinuation = resumeState is not null ? checkpoint.ReadContinuation() : null;
string answersPath = Path.Combine(resultsFolder, $"{(continueStore is null && resumedContinuation is null ? runId : now)}.txt");

// GPU NVIDIA s'il y en a un d'utilisable, sinon CPU. Tout le reste suit ce device :
// le modèle (.to(device)), les batchs (GetBatch) et les tenseurs créés pendant les calculs.
var device = cuda.is_available() ? CUDA : CPU;
Console.WriteLine(device.type == DeviceType.CUDA
    ? "MiniChat - calcul sur : GPU NVIDIA (CUDA)"
    : "MiniChat - calcul sur : CPU (pas de GPU NVIDIA utilisable ; si tu en as un, mets à jour son pilote)");

MiniGpt model;
BpeTokenizer tokenizer;
string? training = null;   // ce qui a été entraîné dans ce lancement (null = modèle juste chargé)

bool newTraining = !resume && (continuePhase is not null || args.Contains("--train") || args.Contains("--finetune") || !store.Exists);
if (newTraining && Checkpoint.ExistsIn(checkpointFolder))
    Console.WriteLine("Attention : un entraînement interrompu existait (reprenable avec --resume). Il est remplacé par ce nouvel entraînement.");

if (resumeState is not null)
{
    // Reprend la phase en cours là où elle s'est arrêtée (poids, optimiseur, pas, meilleure loss).
    (model, tokenizer) = checkpoint.LoadModel(device);
    Console.WriteLine($"Reprise de l'entraînement du {runId} : phase '{resumeState.Phase}', pas {resumeState.Step}.");

    if (resumedContinuation is not null)
    {
        // Un --continue interrompu : mêmes réglages, et toujours rien qu'une phase.
        var continueFrom = resumedContinuation.Phase == "pretrain" ? pretrainStore : store;
        Continue(resumedContinuation, model, tokenizer, pretrainPath, dialoguesPath, continueFrom, continueFrom.ReadInfo(),
            device, log, checkpoint, resumeState);
    }
    else if (resumeState.Phase == "pretrain")
    {
        Pretrain(model, tokenizer, pretrainPath, pretrainStore, TrainingConfig.Pretraining, device, log, checkpoint, resumeState);
        FineTune(model, tokenizer, dialoguesPath, pretrainPath, store, TrainingConfig.FineTuning, device, log, checkpoint, resume: null);
    }
    else
    {
        FineTune(model, tokenizer, dialoguesPath, pretrainPath, store, TrainingConfig.FineTuning, device, log, checkpoint, resumeState);
    }
    training = "entraînement repris (--resume)";
}
else if (continueStore is not null)
{
    // Repart du modèle sauvegardé, avec SON tokenizer : jamais de nouveau BPE, sinon les ids ne voudraient
    // plus rien dire pour le modèle. Les caractères des nouvelles données qu'il ne connaît pas sont ignorés.
    (model, tokenizer) = continueStore.Load(device);
    int steps = (continuePretrainSteps ?? continueFinetuneSteps)!.Value;
    int firstStep = continueInfo?.Step ?? 0;
    var baseConfig = continuePhase == "pretrain" ? TrainingConfig.Pretraining : TrainingConfig.FineTuning;
    var continuation = new ContinueState(continuePhase!, firstStep, steps,
        baseConfig.Continuation(firstStep, steps, continueLearningRate).LearningRate);

    Console.WriteLine($"Modèle chargé depuis '{StoreName(continuePhase!)}'" +
                      (continueInfo is null ? " (sauvegardé avant --continue : pas numérotés depuis 0)." : $", arrêté au pas {firstStep}."));
    checkpoint.Start(model, tokenizer, continuation);
    Continue(continuation, model, tokenizer, pretrainPath, dialoguesPath, continueStore, continueInfo,
        device, log, checkpoint, resume: null);
    training = continuePhase == "pretrain"
        ? $"pré-entraînement continué de {steps} pas (--continue)"
        : $"fine-tuning continué de {steps} pas (--continue-finetune)";
}
else if (args.Contains("--finetune"))
{
    // Refait seulement la phase 2, en repartant du modèle pré-entraîné (pas du modèle final).
    if (!pretrainStore.Exists)
    {
        Console.WriteLine("Pas de modèle pré-entraîné dans 'model-pretrain' : lance d'abord avec --train.");
        return;
    }
    (model, tokenizer) = pretrainStore.Load(device);
    Console.WriteLine("Modèle pré-entraîné chargé depuis 'model-pretrain'.");
    checkpoint.Start(model, tokenizer);
    FineTune(model, tokenizer, dialoguesPath, pretrainPath, store, TrainingConfig.FineTuning, device, log, checkpoint, resume: null);
    training = "fine-tuning seul (--finetune)";
}
else if (!newTraining)
{
    (model, tokenizer) = store.Load(device);
    Console.WriteLine("Modèle chargé depuis le dossier 'model'. (--train pour tout réentraîner, --finetune pour la phase 2 seule, --resume pour reprendre.)");
}
else
{
    (model, tokenizer) = CreateModel(pretrainPath, dialoguesPath, device);
    checkpoint.Start(model, tokenizer);
    Pretrain(model, tokenizer, pretrainPath, pretrainStore, TrainingConfig.Pretraining, device, log, checkpoint, resume: null);
    FineTune(model, tokenizer, dialoguesPath, pretrainPath, store, TrainingConfig.FineTuning, device, log, checkpoint, resume: null);
    training = "pré-entraînement + fine-tuning";
}

// Entraînement terminé jusqu'au bout : plus rien à reprendre.
if (training is not null)
    checkpoint.Delete();

if (device.type == DeviceType.CUDA && GpuMemory.Read() is var (usedMb, totalMb))
    Console.WriteLine($"Mémoire GPU utilisée : {usedMb:N0} / {totalMb:N0} Mo (toute la carte).");

// Après chaque entraînement : le même examen, pour comparer les entraînements entre eux.
if (training is not null)
{
    QuestionTest.Run(new TextGenerator(model, tokenizer, device), questionsPath, answersPath,
        $"Entraînement : {training} | {model.ParameterCount:N0} paramètres | loss : {Path.GetFileName(log.LossCsvPath)}");
}

// Après --continue, le modèle en mémoire est un modèle pré-entraîné : il ne sait pas encore discuter.
if ((continuePhase ?? resumedContinuation?.Phase) == "pretrain")
{
    Console.WriteLine("Pour que le chat profite de ce pré-entraînement : --finetune (refait le fine-tuning depuis 'model-pretrain').");
    if (store.Exists)
        (model, tokenizer) = store.Load(device);
}

RunChat(new ChatSession(new TextGenerator(model, tokenizer, device)));

// ---------------------------------------------------------------------------

// Nouveau modèle : entraîne le tokenizer BPE, puis crée le réseau (poids au hasard).
static (MiniGpt, BpeTokenizer) CreateModel(string pretrainPath, string dialoguesPath, Device device)
{
    manual_seed(TrainingConfig.Pretraining.Seed);

    // Le tokenizer apprend sur les DEUX fichiers : il connaît les mots des dialogues,
    // et le modèle garde le même vocabulaire (donc la même taille de sortie) entre les deux phases.
    Console.WriteLine($"Entraînement du tokenizer BPE ({TokenizerVocabSize} tokens, sur {TokenizerSampleMb} Mo de texte.txt au plus)...");
    var tokenizer = BpeTokenizer.Train(ReadSample(pretrainPath, TokenizerSampleMb) + "\n" + ReadText(dialoguesPath), TokenizerVocabSize);
    var model = new MiniGpt(new ModelConfig { VocabSize = tokenizer.VocabSize }).to(device);

    Console.WriteLine($"Vocabulaire : {tokenizer.VocabSize} tokens ({tokenizer.Alphabet.Length} caractères de base + {tokenizer.Merges.Count} fusions).");
    Console.WriteLine($"Modèle : {model.ParameterCount:N0} paramètres.");
    return (model, tokenizer);
}

// Phase 1 : apprend la langue au modèle sur texte.txt.
// continuation non null (--continue) : store n'est remplacé que si la loss de vérification a baissé.
static void Pretrain(MiniGpt model, BpeTokenizer tokenizer, string pretrainPath, ModelStore store, TrainingConfig config,
    Device device, TrainingLog log, Checkpoint checkpoint, CheckpointState? resume,
    ContinueState? continuation = null, TrainingInfo? previous = null)
{
    // À la reprise, graine décalée du pas : on ne rejoue pas les mêmes extraits qu'au début.
    // Une continuation part aussi de son premier pas (sinon elle retirerait les extraits de --train).
    int seed = config.Seed + (resume?.Step ?? config.FirstStep);
    manual_seed(seed);

    Console.WriteLine("Lecture et découpage en tokens de texte.txt...");
    var dataset = TextDataset.FromFile(CheckExists(pretrainPath), tokenizer, seed);
    Console.WriteLine($"texte.txt : {dataset.Characters:N0} caractères -> {dataset.TrainLength + dataset.ValidationLength:N0} tokens.");
    var result = TrainPhase("Phase 1 - pré-entraînement", "pretrain", dataset, model, tokenizer, config, device, log,
        checkpoint, resume, optimizerPath: continuation is null ? null : store.OptimizerPath);

    SaveResult(store, "model-pretrain", model, tokenizer, result, checkpoint.RunId, continuation, previous);
}

// Phase 2 : spécialise le modèle (déjà pré-entraîné) sur les dialogues.
// Le modèle n'est noté que sur les réponses "IA:", et 20 % de chaque batch vient de texte.txt
// pour qu'il n'oublie pas le français général.
static void FineTune(MiniGpt model, BpeTokenizer tokenizer, string dialoguesPath, string pretrainPath, ModelStore store,
    TrainingConfig config, Device device, TrainingLog log, Checkpoint checkpoint, CheckpointState? resume,
    ContinueState? continuation = null, TrainingInfo? previous = null)
{
    int seed = config.Seed + (resume?.Step ?? config.FirstStep);
    manual_seed(seed);

    var dialogues = new TextDataset(ReadText(dialoguesPath), tokenizer, seed, onlyReplies: true);
    // Autre graine : les extraits de texte général ne suivent pas le même tirage que les dialogues.
    var generalText = TextDataset.FromFile(CheckExists(pretrainPath), tokenizer, seed + 1);

    var result = TrainPhase($"Phase 2 - fine-tuning (loss sur les réponses IA, {config.GeneralTextRatio:P0} de texte général)",
        "finetune", dialogues, model, tokenizer, config, device, log, checkpoint, resume,
        optimizerPath: continuation is null ? null : store.OptimizerPath, generalText);

    SaveResult(store, "model", model, tokenizer, result, checkpoint.RunId, continuation, previous);
}

// --continue / --continue-finetune : steps pas de plus sur un modèle déjà entraîné, avec les données actuelles
// de data/ et l'optimiseur sauvegardé avec lui. Les pas continuent la numérotation (et le CSV) d'avant.
static void Continue(ContinueState continuation, MiniGpt model, BpeTokenizer tokenizer, string pretrainPath,
    string dialoguesPath, ModelStore store, TrainingInfo? previous, Device device, TrainingLog log,
    Checkpoint checkpoint, CheckpointState? resume)
{
    var baseConfig = continuation.Phase == "pretrain" ? TrainingConfig.Pretraining : TrainingConfig.FineTuning;
    var config = baseConfig.Continuation(continuation.FirstStep, continuation.Steps, continuation.LearningRate);
    Console.WriteLine($"Continuation : pas {config.FirstStep} -> {config.Steps}, warmup {config.WarmupSteps} pas, " +
                      $"vitesse {config.LearningRate:0.0e0} -> {config.MinLearningRate:0.0e0}.");

    if (continuation.Phase == "pretrain")
        Pretrain(model, tokenizer, pretrainPath, store, config, device, log, checkpoint, resume, continuation, previous);
    else
        FineTune(model, tokenizer, dialoguesPath, pretrainPath, store, config, device, log, checkpoint, resume, continuation, previous);
}

// Sauvegarde le modèle avec l'état d'AdamW et l'avancement (pas atteint, loss, date du CSV).
// Une continuation ne remplace l'ancien modèle que si sa loss de vérification est meilleure. Les deux loss sont
// mesurées sur les données actuelles (l'ancien modèle au premier pas) : celle notée à l'ancienne sauvegarde
// a pu l'être sur d'autres données, elle n'est affichée que pour info.
static void SaveResult(ModelStore store, string name, MiniGpt model, BpeTokenizer tokenizer, TrainingResult result,
    string runId, ContinueState? continuation, TrainingInfo? previous)
{
    if (continuation is not null)
    {
        string saved = previous is null ? "" : $", {previous.ValLoss:F3} à sa sauvegarde";
        Console.WriteLine($"Loss de vérification : ancienne {result.StartValLoss:F3} (sur les données actuelles{saved}), " +
                          $"nouvelle {result.BestValLoss:F3} (pas {result.BestStep}).");
        if (!result.Improved)
        {
            Console.WriteLine($"Pas meilleure : '{name}' n'est pas remplacé, l'ancien modèle est gardé.");
            return;
        }
    }

    // ponytail: l'état d'AdamW est celui du dernier pas, les poids ceux du meilleur pas ;
    // assez proches pour repartir, pas la peine de sauvegarder l'optimiseur à chaque meilleur pas.
    store.Save(model, tokenizer, result.OptimizerStatePath, new TrainingInfo(result.LastStep, result.BestValLoss, runId));
    Console.WriteLine(continuation is null
        ? $"Modèle sauvegardé dans '{name}'."
        : $"Meilleure : modèle remplacé dans '{name}'.");
}

static TrainingResult TrainPhase(string name, string phase, TextDataset dataset, MiniGpt model, BpeTokenizer tokenizer,
    TrainingConfig config, Device device, TrainingLog log, Checkpoint checkpoint, CheckpointState? resume,
    string? optimizerPath = null, TextDataset? generalText = null)
{
    // Lu dès le début : si le fichier de questions manque, on le sait tout de suite, pas après des heures.
    string sampleQuestion = QuestionTest.ReadQuestions(log.QuestionsPath)[0];
    var generator = new TextGenerator(model, tokenizer, device);

    Console.WriteLine();
    Console.WriteLine($"{name} : {dataset.TrainLength + dataset.ValidationLength:N0} tokens, {config.Steps - config.FirstStep} étapes.");
    return new Trainer(model, dataset, config, device, checkpoint, generalText, phase, log.LossCsvPath,
        showSample: step => QuestionTest.PrintSample(generator, sampleQuestion, step)).Train(resume, optimizerPath);
}

static string StoreName(string phase) => phase == "pretrain" ? "model-pretrain" : "model";

// Lit un fichier de données. Sur Windows les fins de ligne sont "\r\n" : on garde seulement "\n".
static string ReadText(string path) => File.ReadAllText(CheckExists(path)).Replace("\r\n", "\n");

static string CheckExists(string path) =>
    File.Exists(path) ? path : throw new FileNotFoundException($"Fichier de données introuvable : {path}");

// Un fichier de plus de maxMb Mo : maxMb blocs de 1 Mo pris à intervalles réguliers, coupés aux fins de ligne.
// Réparti dans tout le fichier, l'échantillon voit chaque source (texte.txt en colle souvent plusieurs bout à bout).
// Un fichier plus petit est lu en entier.
static string ReadSample(string path, int maxMb)
{
    const int BlockSize = 1 << 20;
    long length = new FileInfo(CheckExists(path)).Length;
    if (length <= (long)maxMb * BlockSize)
        return ReadText(path);

    var sample = new StringBuilder();
    var buffer = new byte[BlockSize];
    using var file = File.OpenRead(path);
    for (int i = 0; i < maxMb; i++)
    {
        file.Position = length / maxMb * i;
        int read = file.ReadAtLeast(buffer, BlockSize, throwOnEndOfStream: false);
        // Le bloc commence et finit au milieu d'une ligne (voire d'un caractère UTF-8) : on garde les lignes entières.
        string block = Encoding.UTF8.GetString(buffer, 0, read);
        int first = block.IndexOf('\n') + 1;
        int last = block.LastIndexOf('\n') + 1;
        if (last > first)
            sample.Append(block, first, last - first);
    }
    return sample.Replace("\r\n", "\n").ToString();
}

static void RunChat(ChatSession chat)
{
    Console.WriteLine();
    Console.WriteLine("Discute avec ton IA. Commandes : /reset (oublier la conversation), /temp 0.8, /topk 20, /topp 0.9, /info, /quit");
    Console.WriteLine();

    while (true)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write(ChatSession.UserPrefix);
        Console.ResetColor();

        string? input = Console.ReadLine();
        if (input is null || input.Trim() == "/quit") break;
        if (string.IsNullOrWhiteSpace(input)) continue;

        if (input.Trim() == "/reset")
        {
            chat.Reset();
            Console.WriteLine("(conversation oubliée)");
            continue;
        }

        if (input.Trim() == "/info")
        {
            Console.WriteLine(chat.Info());
            continue;
        }

        if (TryReadCommand(input, "/temp", out double t))
        {
            chat.Temperature = Math.Clamp(t, 0.1, 2.0);
            Console.WriteLine($"(température = {chat.Temperature})");
            continue;
        }

        if (TryReadCommand(input, "/topk", out double k))
        {
            chat.TopK = Math.Max(0, (int)k);
            Console.WriteLine(chat.TopK == 0 ? "(top-k désactivé)" : $"(top-k = {chat.TopK})");
            continue;
        }

        if (TryReadCommand(input, "/topp", out double p))
        {
            chat.TopP = Math.Clamp(p, 0.05, 1.0);
            Console.WriteLine(chat.TopP >= 1.0 ? "(top-p désactivé)" : $"(top-p = {chat.TopP})");
            continue;
        }

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.Write(ChatSession.BotPrefix);
        chat.Send(input, Console.Write);
        Console.ResetColor();
        Console.WriteLine();
    }
}

// Lit une commande du chat avec un nombre, ex. "/temp 0,8" ou "/temp 0.8" (virgule ou point acceptés).
static bool TryReadCommand(string input, string command, out double value)
{
    value = 0;
    return input.StartsWith(command + " ")
        && double.TryParse(input[(command.Length + 1)..].Replace(',', '.'),
            System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out value);
}

// Valeur d'une option "--nom valeur", ou null si l'option n'est pas là. example : montré si la valeur manque.
static string? ReadOption(string[] args, string name, string example)
{
    int i = Array.IndexOf(args, name);
    if (i < 0) return null;
    if (i + 1 >= args.Length || args[i + 1].StartsWith("--"))
    {
        Console.WriteLine($"{name} attend une valeur, par exemple : {name} {example}");
        Environment.Exit(1);
    }
    return args[i + 1];
}

// Nombre positif d'une option, ex. "--lr 3e-4" (virgule ou point acceptés), ou null si l'option n'est pas là.
static double? ReadNumber(string[] args, string name, string example)
{
    string? text = ReadOption(args, name, example);
    if (text is null) return null;
    if (double.TryParse(text.Replace(',', '.'), System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out double value) && value > 0)
        return value;

    Console.WriteLine($"{name} attend un nombre positif, par exemple : {name} {example}");
    Environment.Exit(1);
    return null;
}

// Nombre de pas d'une option, ex. "--continue 2000", ou null si l'option n'est pas là.
static int? ReadSteps(string[] args, string name)
{
    if (ReadNumber(args, name, "2000") is not { } value) return null;
    if (value == Math.Floor(value) && value <= int.MaxValue / 2) return (int)value;

    Console.WriteLine($"{name} attend un nombre entier de pas, par exemple : {name} 2000");
    Environment.Exit(1);
    return null;
}

/// <summary>Où un entraînement écrit son suivi : le CSV des loss, et les questions pour les exemples.</summary>
record TrainingLog(string LossCsvPath, string QuestionsPath);
