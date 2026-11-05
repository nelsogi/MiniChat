using System.Text;
using MiniChat.Chat;
using MiniChat.Config;
using MiniChat.Data;
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
const int TokenizerVocabSize = 2000;

// Chemins à côté de l'exécutable : marche avec "dotnet run" comme avec F5 dans Visual Studio.
string pretrainPath = Path.Combine(AppContext.BaseDirectory, "data", "texte.txt");
string dialoguesPath = Path.Combine(AppContext.BaseDirectory, "data", "dialogues.txt");

// GPU NVIDIA s'il y en a un d'utilisable, sinon CPU. Tout le reste suit ce device :
// le modèle (.to(device)), les batchs (GetBatch) et les tenseurs créés pendant les calculs.
var device = cuda.is_available() ? CUDA : CPU;
Console.WriteLine(device.type == DeviceType.CUDA
    ? "MiniChat - calcul sur : GPU NVIDIA (CUDA)"
    : "MiniChat - calcul sur : CPU (pas de GPU NVIDIA utilisable ; si tu en as un, mets à jour son pilote)");

// Deux sauvegardes : le modèle pré-entraîné (base pour refaire le fine-tuning) et le modèle final.
var pretrainStore = new ModelStore(Path.Combine(AppContext.BaseDirectory, "model-pretrain"));
var store = new ModelStore(Path.Combine(AppContext.BaseDirectory, "model"));

MiniGpt model;
BpeTokenizer tokenizer;

if (args.Contains("--finetune"))
{
    // Refait seulement la phase 2, en repartant du modèle pré-entraîné (pas du modèle final).
    if (!pretrainStore.Exists)
    {
        Console.WriteLine("Pas de modèle pré-entraîné dans 'model-pretrain' : lance d'abord avec --train.");
        return;
    }
    (model, tokenizer) = pretrainStore.Load(device);
    Console.WriteLine("Modèle pré-entraîné chargé depuis 'model-pretrain'.");
    FineTune(model, tokenizer, dialoguesPath, pretrainPath, store, device);
}
else if (store.Exists && !args.Contains("--train"))
{
    (model, tokenizer) = store.Load(device);
    Console.WriteLine("Modèle chargé depuis le dossier 'model'. (--train pour tout réentraîner, --finetune pour la phase 2 seule.)");
}
else
{
    (model, tokenizer) = Pretrain(pretrainPath, dialoguesPath, pretrainStore, device);
    FineTune(model, tokenizer, dialoguesPath, pretrainPath, store, device);
}

RunChat(new ChatSession(new TextGenerator(model, tokenizer, device)));

// ---------------------------------------------------------------------------

// Phase 1 : crée un nouveau modèle et lui apprend la langue sur texte.txt.
static (MiniGpt, BpeTokenizer) Pretrain(string pretrainPath, string dialoguesPath, ModelStore store, Device device)
{
    var config = TrainingConfig.Pretraining;
    manual_seed(config.Seed);

    string text = ReadText(pretrainPath);

    // Le tokenizer apprend sur les DEUX fichiers : il connaît les mots des dialogues,
    // et le modèle garde le même vocabulaire (donc la même taille de sortie) entre les deux phases.
    Console.WriteLine($"Entraînement du tokenizer BPE ({TokenizerVocabSize} tokens)...");
    var tokenizer = BpeTokenizer.Train(text + "\n" + ReadText(dialoguesPath), TokenizerVocabSize);
    var model = new MiniGpt(new ModelConfig { VocabSize = tokenizer.VocabSize }).to(device);

    Console.WriteLine($"Vocabulaire : {tokenizer.VocabSize} tokens ({tokenizer.Alphabet.Length} caractères de base + {tokenizer.Merges.Count} fusions).");
    Console.WriteLine($"Modèle : {model.ParameterCount:N0} paramètres.");
    var dataset = new TextDataset(text, tokenizer, config.Seed);
    Console.WriteLine($"texte.txt : {text.Length:N0} caractères -> {dataset.TrainLength + dataset.ValidationLength:N0} tokens.");
    TrainPhase("Phase 1 - pré-entraînement", dataset, model, config, device);

    store.Save(model, tokenizer);
    Console.WriteLine("Modèle pré-entraîné sauvegardé dans 'model-pretrain'.");
    return (model, tokenizer);
}

// Phase 2 : spécialise le modèle (déjà pré-entraîné) sur les dialogues.
// Le modèle n'est noté que sur les réponses "IA:", et 20 % de chaque batch vient de texte.txt
// pour qu'il n'oublie pas le français général.
static void FineTune(MiniGpt model, BpeTokenizer tokenizer, string dialoguesPath, string pretrainPath, ModelStore store, Device device)
{
    var config = TrainingConfig.FineTuning;
    manual_seed(config.Seed);

    var dialogues = new TextDataset(ReadText(dialoguesPath), tokenizer, config.Seed, onlyReplies: true);
    // Autre graine : les extraits de texte général ne suivent pas le même tirage que les dialogues.
    var generalText = new TextDataset(ReadText(pretrainPath), tokenizer, config.Seed + 1);

    TrainPhase($"Phase 2 - fine-tuning (loss sur les réponses IA, {config.GeneralTextRatio:P0} de texte général)",
        dialogues, model, config, device, generalText);

    store.Save(model, tokenizer);
    Console.WriteLine("Modèle final sauvegardé dans 'model'.");
}

static void TrainPhase(string name, TextDataset dataset, MiniGpt model, TrainingConfig config, Device device, TextDataset? generalText = null)
{
    Console.WriteLine();
    Console.WriteLine($"{name} : {dataset.TrainLength + dataset.ValidationLength:N0} tokens, {config.Steps} étapes.");
    new Trainer(model, dataset, config, device, generalText).Train();
}

// Lit un fichier de données. Sur Windows les fins de ligne sont "\r\n" : on garde seulement "\n".
static string ReadText(string path)
{
    if (!File.Exists(path))
        throw new FileNotFoundException($"Fichier de données introuvable : {path}");
    return File.ReadAllText(path).Replace("\r\n", "\n");
}

static void RunChat(ChatSession chat)
{
    Console.WriteLine();
    Console.WriteLine("Discute avec ton IA. Commandes : /reset (oublier la conversation), /temp 0.8, /topk 20, /topp 0.9, /quit");
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
