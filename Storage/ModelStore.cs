using System.Text.Json;
using MiniChat.Config;
using MiniChat.Model;
using MiniChat.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace MiniChat.Storage;

/// <summary>
/// Ce qui a produit les poids sauvegardés. Step = pas atteint à la fin (une continuation repart de là),
/// ValLoss = loss de vérification des poids gardés, RunId = date de l'entraînement (nom de son CSV de loss).
/// </summary>
public sealed record TrainingInfo(int Step, float ValLoss, string RunId);

/// <summary>
/// Sauvegarde et recharge un modèle entraîné dans un dossier :
///   config.json     -> la forme du réseau
///   tokenizer.json  -> le tokenizer BPE (caractères de base + fusions apprises)
///   weights.dat     -> les poids appris
///   optimizer.dat   -> l'état d'AdamW à la fin de l'entraînement (pour --continue)
///   training.json   -> pas atteint, loss de vérification, date de l'entraînement (voir TrainingInfo)
/// Comme ça, pas besoin de réentraîner à chaque lancement.
/// Les dossiers d'avant --continue n'ont ni optimizer.dat ni training.json : ils restent utilisables.
///
/// Un ancien dossier (avec vocab.txt, du temps du tokenizer caractère par caractère) n'est pas
/// reconnu : Exists renvoie false et le programme réentraîne.
/// </summary>
public sealed class ModelStore
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    /// <summary>Forme du fichier tokenizer.json. Merges[i] = [gauche, droite] de la fusion n° i.</summary>
    private sealed record TokenizerFile(string Alphabet, int[][] Merges);

    private readonly string _folder;

    public ModelStore(string folder) => _folder = folder;

    private string ConfigPath => Path.Combine(_folder, "config.json");
    private string TokenizerPath => Path.Combine(_folder, "tokenizer.json");
    public string WeightsPath => Path.Combine(_folder, "weights.dat");
    public string OptimizerPath => Path.Combine(_folder, "optimizer.dat");
    private string InfoPath => Path.Combine(_folder, "training.json");

    public bool Exists => File.Exists(ConfigPath) && File.Exists(TokenizerPath) && File.Exists(WeightsPath);

    public string Folder => _folder;

    /// <summary>Quand les poids ont été écrits (fin de l'entraînement, ou /save).</summary>
    public DateTime SavedAt => File.GetLastWriteTime(WeightsPath);

    /// <summary>Taille des poids sur disque, en octets.</summary>
    public long WeightsBytes => new FileInfo(WeightsPath).Length;

    /// <summary>La forme du réseau, sans charger les poids (rapide : pour lister les modèles).</summary>
    public ModelConfig ReadConfig() =>
        JsonSerializer.Deserialize<ModelConfig>(File.ReadAllText(ConfigPath)) ?? throw new InvalidDataException("config.json illisible.");

    /// <param name="optimizerStatePath">Fichier d'état d'AdamW à copier avec le modèle (null = pas d'optimizer.dat).</param>
    /// <param name="info">Avancement à garder avec le modèle (null = pas de training.json).</param>
    public void Save(MiniGpt model, BpeTokenizer tokenizer, string? optimizerStatePath = null, TrainingInfo? info = null)
    {
        Directory.CreateDirectory(_folder);
        File.WriteAllText(ConfigPath, JsonSerializer.Serialize(model.Config, JsonOptions));

        var merges = tokenizer.Merges.Select(m => new[] { m.Left, m.Right }).ToArray();
        File.WriteAllText(TokenizerPath, JsonSerializer.Serialize(new TokenizerFile(tokenizer.Alphabet, merges)));

        model.save(WeightsPath);

        if (optimizerStatePath is not null)
            File.Copy(optimizerStatePath, OptimizerPath, overwrite: true);
        if (info is not null)
            File.WriteAllText(InfoPath, JsonSerializer.Serialize(info, JsonOptions));
    }

    /// <summary>L'avancement sauvegardé avec le modèle, ou null pour un modèle d'avant --continue.</summary>
    public TrainingInfo? ReadInfo() =>
        File.Exists(InfoPath) ? JsonSerializer.Deserialize<TrainingInfo>(File.ReadAllText(InfoPath)) : null;

    public (MiniGpt Model, BpeTokenizer Tokenizer) Load(Device device)
    {
        var config = ReadConfig();

        var file = JsonSerializer.Deserialize<TokenizerFile>(File.ReadAllText(TokenizerPath))
                   ?? throw new InvalidDataException("tokenizer.json illisible.");
        var tokenizer = new BpeTokenizer(file.Alphabet, file.Merges.Select(m => (m[0], m[1])));

        var model = new MiniGpt(config);
        model.load(WeightsPath);
        return (model.to(device), tokenizer);
    }
}
