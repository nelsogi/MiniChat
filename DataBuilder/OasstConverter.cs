using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace DataBuilder;

/// <summary>
/// Convertit OpenAssistant (fichier "*.messages.jsonl" ou ".jsonl.gz", un message JSON par ligne)
/// au format de MiniChat (voir DialogueConverter).
///
/// Dans OpenAssistant, les conversations sont des arbres : chaque message peut avoir plusieurs
/// réponses, classées par des humains ("rank", 0 = la meilleure). On part de chaque question
/// de départ et on suit toujours la meilleure réponse, ce qui donne une conversation par arbre.
/// </summary>
public sealed class OasstConverter : DialogueConverter
{
    private sealed record Message(string Id, string? ParentId, string Role, string Text, int Rank);

    private readonly string _lang;

    /// <param name="lang">Langue gardée ("fr", "en"...).</param>
    /// <param name="maxLength">Longueur max d'un message.</param>
    /// <param name="maxSentences">Nombre max de phrases gardées dans une réponse de l'IA.</param>
    public OasstConverter(string lang, int maxLength, int maxSentences) : base(maxLength, maxSentences)
    {
        _lang = lang;
    }

    public override string Convert(IEnumerable<string> inputPaths)
    {
        var messages = inputPaths.SelectMany(ReadMessages).ToList();
        var replies = messages.Where(m => m.ParentId is not null).ToLookup(m => m.ParentId!);

        foreach (var root in messages.Where(m => m.ParentId is null))
        {
            var (lines, stopReason) = FollowBestBranch(root, replies);
            Add(lines, stopReason);
        }

        return Output;
    }

    /// <summary>
    /// Descend l'arbre en prenant à chaque fois la réponse la mieux classée.
    /// StopReason = pourquoi on s'est arrêté avant la fin de l'arbre (null si on est allé au bout).
    /// </summary>
    private (List<string> Lines, string? StopReason) FollowBestBranch(Message root, ILookup<string, Message> replies)
    {
        var lines = new List<string>();
        string? stopReason = null;
        Message? current = root;

        while (current is not null)
        {
            bool fromUser = current.Role == "prompter";
            var (text, rejection) = Clean(current.Text, fromUser);
            if (text is null)
            {
                stopReason = rejection;   // message inutilisable : la conversation s'arrête là
                break;
            }

            lines.Add((fromUser ? "Toi: " : "IA: ") + text);

            current = replies[current.Id].OrderBy(m => m.Rank).FirstOrDefault();
        }

        return (lines, stopReason);
    }

    /// <summary>Lit les messages de la bonne langue, en écartant ceux supprimés ou refusés par les relecteurs.</summary>
    private IEnumerable<Message> ReadMessages(string path)
    {
        using var file = File.OpenRead(path);
        using Stream stream = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase)
            ? new GZipStream(file, CompressionMode.Decompress)
            : file;
        using var reader = new StreamReader(stream, Encoding.UTF8);

        while (reader.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            using var json = JsonDocument.Parse(line);
            var m = json.RootElement;

            if (GetString(m, "lang") != _lang) continue;
            if (GetBool(m, "deleted") == true) { Count(IgnoredMessages, "supprimés"); continue; }
            if (GetBool(m, "review_result") == false) { Count(IgnoredMessages, "refusés par les relecteurs"); continue; }

            yield return new Message(
                Id: GetString(m, "message_id") ?? throw new InvalidDataException($"message sans message_id dans {path}"),
                ParentId: GetString(m, "parent_id"),
                Role: GetString(m, "role") ?? "",
                Text: GetString(m, "text") ?? "",
                // Pas de rang = pas classé : on le met après les réponses classées.
                Rank: m.TryGetProperty("rank", out var r) && r.ValueKind == JsonValueKind.Number ? r.GetInt32() : int.MaxValue);
        }
    }

    private static string? GetString(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static bool? GetBool(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : null;
}
