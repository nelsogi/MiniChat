using MiniChat.Chat;
using MiniChat.Config;
using MiniChat.Data;
using MiniChat.Model;
using MiniChat.Tokenization;
using MiniChat.Training;
using Xunit;
using static TorchSharp.torch;

// Les tests partagent le tokenizer de TestData (son cache d'encodage n'est pas fait pour plusieurs threads) et la graine
// globale de TorchSharp (manual_seed) : ils tournent un par un. Tous ensemble, ~1 s.
[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace MiniChat.Tests;

/// <summary>Ce que les tests partagent : un petit texte, un tokenizer appris dessus, un tout petit modèle sur CPU.</summary>
internal static class TestData
{
    public const string Text =
        "Le chat dort sur le canapé. Le chien joue dans le jardin avec une balle rouge.\n" +
        "Les enfants vont à l'école le matin, et ils rentrent le soir à 17 h.\n" +
        "Paris est la capitale de la France ; Lyon et Marseille sont de grandes villes.\n" +
        "Bonjour ! Comment vas-tu ? Très bien, merci, et toi ?\n";

    public const string Dialogues =
        "Toi: Bonjour !\nIA: Bonjour, comment vas-tu ?\nToi: Très bien, merci.\nIA: Tant mieux !\n\n" +
        "Toi: Quelle est la capitale de la France ?\nIA: Paris est la capitale de la France.\n";

    private static readonly Lazy<BpeTokenizer> SharedTokenizer = new(() =>
        BpeTokenizer.Train(Text + "\n" + ChatFormat.FromDialogueFile(Dialogues), 120));

    /// <summary>Un tokenizer de 120 tokens appris sur Text et Dialogues (toujours le même).</summary>
    public static BpeTokenizer Tokenizer => SharedTokenizer.Value;

    /// <summary>Un modèle minuscule : 1 couche, 32 dimensions, contexte de 16 tokens.</summary>
    public static ModelConfig TinyConfig(int vocabSize, ModelArchitecture architecture = ModelArchitecture.Modern) =>
        new() { VocabSize = vocabSize, BlockSize = 16, EmbedDim = 32, Heads = 2, Layers = 1, Architecture = architecture };

    public static MiniGpt TinyModel(int seed = 1, ModelArchitecture architecture = ModelArchitecture.Modern)
    {
        manual_seed(seed);
        return new MiniGpt(TinyConfig(Tokenizer.VocabSize, architecture));
    }

    /// <summary>length tokens tirés au hasard (graine fixe) dans le vocabulaire de Tokenizer.</summary>
    public static TokenizedText RandomTokens(int length, int seed = 3)
    {
        var random = new Random(seed);
        var ids = new short[length];
        for (int i = 0; i < length; i++)
            ids[i] = (short)random.Next(Tokenizer.VocabSize);
        return new TokenizedText(ids, null, length);
    }

    /// <summary>
    /// count conversations de 1 à 3 tours, de longueurs variées (graine fixe), au format du modèle, découpées
    /// en tokens avec le masque des réponses (comme dialogues.txt au fine-tuning).
    /// </summary>
    public static TokenizedText Conversations(int count, int seed = 5) =>
        TextDataset.Tokenize(ChatFormat.FromDialogueFile(DialogueFile(count, seed)), Tokenizer, onlyReplies: true);

    /// <summary>Les mêmes conversations, au format de dialogues.txt (Toi: / IA:, une ligne vide entre deux).</summary>
    public static string DialogueFile(int count, int seed = 5)
    {
        string[] questions = ["Bonjour !", "Comment vas-tu ?", "Quelle est la capitale de la France ?", "Le chat dort ?"];
        string[] replies = ["Bien.", "Paris est la capitale de la France.", "Le chien joue dans le jardin avec une balle rouge.",
            "Très bien, merci, et toi ?", "Oui."];
        var random = new Random(seed);
        return string.Join("\n\n", Enumerable.Range(0, count).Select(_ => string.Join("\n",
            Enumerable.Range(0, random.Next(1, 4)).Select(_ =>
                $"Toi: {questions[random.Next(questions.Length)]}\nIA: {replies[random.Next(replies.Length)]}"))));
    }

    /// <summary>Réglages d'un entraînement de quelques pas sur le modèle minuscule.</summary>
    public static TrainingConfig TinyTraining(int steps) => new()
    {
        Epochs = null,
        Steps = steps,
        BatchSize = 4,
        EffectiveBatchSize = 8,
        WarmupSteps = 2,
        EvalInterval = 5,
        EvalBatches = 2,
        ValidationExtracts = 4,
        CheckpointInterval = 1000,
        SampleInterval = 1000,
        EarlyStoppingPatience = 0,
    };
}

/// <summary>Un dossier temporaire, supprimé à la fin du test.</summary>
internal sealed class TempFolder : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "minichat-tests-" + Guid.NewGuid().ToString("N"));

    public TempFolder() => Directory.CreateDirectory(Path);

    public string File(string name) => System.IO.Path.Combine(Path, name);

    public void Dispose()
    {
        if (Directory.Exists(Path))
            Directory.Delete(Path, recursive: true);
    }
}

/// <summary>Ce que le Trainer montre, gardé pour les vérifications. onAdvance : appelé avant chaque pas.</summary>
internal sealed class RecordingProgress(Action<int>? onAdvance = null) : ITrainingProgress
{
    public List<(int Step, double LearningRate)> Steps { get; } = [];
    public List<TrainingReport> Reports { get; } = [];
    public List<string> Messages { get; } = [];

    public void Advance(int step, double learningRate, TimeSpan? remaining)
    {
        Steps.Add((step, learningRate));
        onAdvance?.Invoke(step);
    }

    public void Report(TrainingReport report) => Reports.Add(report);

    public void Log(string message) => Messages.Add(message);
}
