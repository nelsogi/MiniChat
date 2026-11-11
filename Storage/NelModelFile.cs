using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using MiniChat.Config;
using MiniChat.Model;
using MiniChat.Tokenization;
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
/// Un fichier .nelmodel (voir NELMODEL-FORMAT.md) : un zip avec, à la racine,
///   config.json, tokenizer.json, weights.dat   copiés du dossier de modèle (ce que lit Nel Code, comme en version 1)
///   model.safetensors                          les mêmes poids, lisibles sans TorchSharp (version 2)
///   reference.json                             ids et sorties de référence du modèle (version 2)
///   model-info.json                            la description
/// Pas d'optimizer.dat : un .nelmodel sert à générer.
/// </summary>
public static class NelModelFile
{
    public const string Extension = ".nelmodel";
    public const int FormatVersion = 2;

    private const string InfoEntry = "model-info.json";
    private const string SafeTensorsEntry = "model.safetensors";

    // Fichiers calculés à l'export (pas dans le dossier de modèle), écrits par WriteGenerated.
    private static readonly string[] GeneratedEntries = { SafeTensorsEntry, NelModelReference.FileName };

    private static readonly JsonSerializerOptions InfoOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,   // "Pré-entraîné", pas "Pré..."
    };

    /// <summary>Écrit model.safetensors et reference.json de ce modèle (chargé sur le CPU) dans folder.</summary>
    public static void WriteGenerated(MiniGpt model, BpeTokenizer tokenizer, string folder)
    {
        SafeTensors.Write(model, Path.Combine(folder, SafeTensorsEntry), new Dictionary<string, string>
        {
            ["format"] = "pt",   // la convention de safetensors pour des poids PyTorch / TorchSharp
            ["nelmodel.formatVersion"] = FormatVersion.ToString(),
            ["architecture"] = model.Config.Architecture.ToString(),
        });
        NelModelReference.Write(model, tokenizer, Path.Combine(folder, NelModelReference.FileName));
    }

    /// <summary>
    /// Écrit le .nelmodel dans un .tmp, puis le renomme : un export coupé en route ne laisse pas de fichier à moitié écrit.
    /// generated : le dossier rempli par WriteGenerated.
    /// </summary>
    public static void Write(ModelStore store, string generated, NelModelInfo info, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string tmp = path + ".tmp";
        using (var zip = ZipFile.Open(tmp, ZipArchiveMode.Create))
        {
            foreach (var (entry, source) in ModelFiles(store).Concat(GeneratedEntries.Select(e => (e, Path.Combine(generated, e)))))
                // Les poids (float32) ne se compressent pas : mesuré, Fastest les grossit de 5 % (111 -> 117 Mo).
                zip.CreateEntryFromFile(source, entry, IsWeights(entry) ? CompressionLevel.NoCompression : CompressionLevel.Optimal);

            using var writer = new StreamWriter(zip.CreateEntry(InfoEntry).Open());
            writer.Write(JsonSerializer.Serialize(info, InfoOptions));
        }
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>
    /// Relit un .nelmodel qu'on vient d'écrire, comme le ferait /import de Nel Code : les six fichiers à la racine,
    /// pas d'optimizer.dat, chaque fichier du modèle identique à l'original, model-info.json lisible (bonne
    /// formatVersion, bon nombre de paramètres), assez de tokens pour VocabSize, les poids qui se rechargent,
    /// model.safetensors identique à weights.dat, et les ids de reference.json redonnés par tokenizer.json.
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
                var expected = ModelFiles(source).Select(f => f.Entry).Concat(GeneratedEntries).Append(InfoEntry).ToList();
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
                SafeTensors.Verify(Path.Combine(folder, SafeTensorsEntry), model);
            }
            VerifyReference(Path.Combine(folder, NelModelReference.FileName), tokenizer);
        }
        finally
        {
            if (Directory.Exists(folder))
                Directory.Delete(folder, recursive: true);
        }
    }

    // Les textes de reference.json redécoupés avec le tokenizer.json de l'archive : mêmes ids.
    private static void VerifyReference(string path, BpeTokenizer tokenizer)
    {
        var reference = JsonNode.Parse(File.ReadAllText(path)) ?? throw new InvalidDataException($"{NelModelReference.FileName} illisible.");
        var texts = reference["texts"]?.AsArray() ?? throw new InvalidDataException($"{NelModelReference.FileName} : pas de textes.");
        foreach (var text in texts)
        {
            long[] ids = text!["ids"]!.AsArray().Select(id => (long)id!).ToArray();
            if (!tokenizer.Encode((string)text["text"]!).SequenceEqual(ids))
                throw new InvalidDataException($"{NelModelReference.FileName} : les ids de « {text["text"]} » ne sont pas ceux du tokenizer.");
        }
        if (reference["chat"]?.AsArray().Count != NelModelReference.ChatMessages.Length)
            throw new InvalidDataException($"{NelModelReference.FileName} : il manque des prompts de chat.");
    }

    private static bool IsWeights(string entry) => entry is "weights.dat" or SafeTensorsEntry;

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
