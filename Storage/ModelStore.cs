using System.Text.Json;
using MiniChat.Config;
using MiniChat.Model;
using MiniChat.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace MiniChat.Storage;

/// <summary>
/// Sauvegarde et recharge un modèle entraîné dans un dossier :
///   config.json     -> la forme du réseau
///   tokenizer.json  -> le tokenizer BPE (caractères de base + fusions apprises)
///   weights.dat     -> les poids appris
/// Comme ça, pas besoin de réentraîner à chaque lancement.
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
    private string WeightsPath => Path.Combine(_folder, "weights.dat");

    public bool Exists => File.Exists(ConfigPath) && File.Exists(TokenizerPath) && File.Exists(WeightsPath);

    public void Save(MiniGpt model, BpeTokenizer tokenizer)
    {
        Directory.CreateDirectory(_folder);
        File.WriteAllText(ConfigPath, JsonSerializer.Serialize(model.Config, JsonOptions));

        var merges = tokenizer.Merges.Select(m => new[] { m.Left, m.Right }).ToArray();
        File.WriteAllText(TokenizerPath, JsonSerializer.Serialize(new TokenizerFile(tokenizer.Alphabet, merges)));

        model.save(WeightsPath);
    }

    public (MiniGpt Model, BpeTokenizer Tokenizer) Load(Device device)
    {
        var config = JsonSerializer.Deserialize<ModelConfig>(File.ReadAllText(ConfigPath))
                     ?? throw new InvalidDataException("config.json illisible.");

        var file = JsonSerializer.Deserialize<TokenizerFile>(File.ReadAllText(TokenizerPath))
                   ?? throw new InvalidDataException("tokenizer.json illisible.");
        var tokenizer = new BpeTokenizer(file.Alphabet, file.Merges.Select(m => (m[0], m[1])));

        var model = new MiniGpt(config);
        model.load(WeightsPath);
        return (model.to(device), tokenizer);
    }
}
