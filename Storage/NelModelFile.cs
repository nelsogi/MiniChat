using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using MiniChat.Config;
using static TorchSharp.torch;

namespace MiniChat.Storage;

/// <summary>
/// Description d'un modèle exporté (model-info.json, voir NELMODEL-FORMAT.md). Seul FormatVersion est obligatoire :
/// les clés à null ne sont pas écrites.
/// </summary>
public sealed record NelModelInfo(
    [property: JsonPropertyName("formatVersion")] int FormatVersion,
    [property: JsonPropertyName("name")] string? Name,
    [property: JsonPropertyName("createdAt")] string? CreatedAt,
    [property: JsonPropertyName("parameters")] long? Parameters,
    [property: JsonPropertyName("architecture")] ModelArchitecture? Architecture,
    [property: JsonPropertyName("valLoss")] float? ValLoss,
    [property: JsonPropertyName("trainingSteps")] int? TrainingSteps,
    [property: JsonPropertyName("description")] string? Description);

/// <summary>
/// Un fichier .nelmodel (voir NELMODEL-FORMAT.md) : un zip avec, à la racine, config.json, tokenizer.json et
/// weights.dat d'un dossier de modèle, plus model-info.json. Pas d'optimizer.dat : un .nelmodel sert à générer.
/// </summary>
public static class NelModelFile
{
    public const string Extension = ".nelmodel";
    public const int FormatVersion = 1;

    private const string InfoEntry = "model-info.json";

    private static readonly JsonSerializerOptions InfoOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,   // "Pré-entraîné", pas "Pré..."
    };

    /// <summary>
    /// Écrit le .nelmodel dans un .tmp, puis le renomme : un export coupé en route ne laisse pas de fichier à moitié écrit.
    /// </summary>
    public static void Write(ModelStore store, NelModelInfo info, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string tmp = path + ".tmp";
        using (var zip = ZipFile.Open(tmp, ZipArchiveMode.Create))
        {
            foreach (var (entry, source) in ModelFiles(store))
                // Les poids (float32) ne se compressent pas : mesuré, Fastest les grossit de 5 % (111 -> 117 Mo).
                zip.CreateEntryFromFile(source, entry, entry == "weights.dat" ? CompressionLevel.NoCompression : CompressionLevel.Optimal);

            using var writer = new StreamWriter(zip.CreateEntry(InfoEntry).Open());
            writer.Write(JsonSerializer.Serialize(info, InfoOptions));
        }
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>
    /// Relit un .nelmodel qu'on vient d'écrire, comme le ferait /import de Nel Code : les quatre fichiers à la racine,
    /// pas d'optimizer.dat, chaque fichier identique à celui du modèle, model-info.json lisible (formatVersion 1,
    /// bon nombre de paramètres), assez de tokens pour VocabSize, et les poids qui se rechargent.
    /// Lance InvalidDataException au premier problème.
    /// </summary>
    public static void Verify(string path, ModelStore source, long parameters)
    {
        string folder = Path.Combine(Path.GetTempPath(), "minichat-export-" + Guid.NewGuid().ToString("N"));
        try
        {
            using (var zip = ZipFile.OpenRead(path))
            {
                var names = zip.Entries.Select(e => e.FullName).ToList();
                var expected = ModelFiles(source).Select(f => f.Entry).Append(InfoEntry).ToList();
                if (expected.Except(names).ToList() is { Count: > 0 } missing)
                    throw new InvalidDataException($"il manque {string.Join(", ", missing)}.");
                if (names.Except(expected).ToList() is { Count: > 0 } extra)
                    throw new InvalidDataException($"fichiers en trop : {string.Join(", ", extra)}.");

                Directory.CreateDirectory(folder);
                foreach (var entry in zip.Entries)
                    entry.ExtractToFile(Path.Combine(folder, entry.FullName));
            }

            foreach (var (entry, file) in ModelFiles(source))
                if (!SameBytes(Path.Combine(folder, entry), file))
                    throw new InvalidDataException($"{entry} n'est pas identique à celui du modèle.");

            var info = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, InfoEntry)))
                       ?? throw new InvalidDataException($"{InfoEntry} illisible.");
            if ((int?)info["formatVersion"] != FormatVersion)
                throw new InvalidDataException($"{InfoEntry} : formatVersion devrait valoir {FormatVersion}.");

            var store = new ModelStore(folder);
            var config = store.ReadConfig();
            var (model, tokenizer) = store.Load(CPU);   // chargement strict : poids manquant, en trop ou mal formé = erreur
            using (model)
            {
                if (tokenizer.VocabSize != config.VocabSize)
                    throw new InvalidDataException($"tokenizer.json a {tokenizer.VocabSize} tokens, config.json en attend {config.VocabSize}.");
                if (model.ParameterCount != parameters || (long?)info["parameters"] != parameters)
                    throw new InvalidDataException($"{InfoEntry} annonce {info["parameters"]} paramètres, le modèle en a {model.ParameterCount}.");
            }
        }
        finally
        {
            if (Directory.Exists(folder))
                Directory.Delete(folder, recursive: true);
        }
    }

    // Nom dans le zip -> fichier du dossier de modèle. Pas optimizer.dat ni training.json.
    private static IEnumerable<(string Entry, string Source)> ModelFiles(ModelStore store) =>
        new[] { store.ConfigPath, store.TokenizerPath, store.WeightsPath }.Select(p => (Path.GetFileName(p), p));

    private static bool SameBytes(string a, string b)
    {
        using var streamA = File.OpenRead(a);
        using var streamB = File.OpenRead(b);
        return streamA.Length == streamB.Length && SHA256.HashData(streamA).AsSpan().SequenceEqual(SHA256.HashData(streamB));
    }
}
