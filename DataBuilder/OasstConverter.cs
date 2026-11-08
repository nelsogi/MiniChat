using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace DataBuilder;

/// <summary>
/// Convertit OpenAssistant (fichier "*.messages.jsonl" ou ".jsonl.gz", un message JSON par ligne)
/// au format de MiniChat :
///     Toi: question
///     IA: réponse
///                          <- une ligne vide entre deux conversations
///     Toi: question
///
/// Dans OpenAssistant, les conversations sont des arbres : chaque message peut avoir plusieurs
/// réponses, classées par des humains ("rank", 0 = la meilleure). On part de chaque question
/// de départ et on suit toujours la meilleure réponse, ce qui donne une conversation par arbre.
///
/// Après Convert(), les compteurs (Kept, Rejected...) disent ce qui a été gardé ou rejeté, et pourquoi.
/// </summary>
public sealed class OasstConverter
{
    private sealed record Message(string Id, string? ParentId, string Role, string Text, int Rank);

    // Raisons de rejet, affichées telles quelles dans le bilan.
    private const string NoReply = "pas de réponse utilisable";
    private const string HasCode = "contient du code";
    private const string ExoticChars = "caractères exotiques (emoji...)";
    private const string QuestionTooLong = "question trop longue";
    private const string NoSentenceEnd = "réponse trop longue, sans fin de phrase où couper";
    private const string Empty = "message vide";

    private readonly string _lang;
    private readonly int _maxLength;
    private readonly int _maxSentences;

    /// <summary>Réponses de l'IA raccourcies à maxSentences phrases.</summary>
    public int LimitedReplies { get; private set; }

    /// <summary>Conversations écrites dans le fichier.</summary>
    public int Kept { get; private set; }

    /// <summary>Conversations gardées mais arrêtées avant la fin (un message plus loin était inutilisable).</summary>
    public int Shortened { get; private set; }

    /// <summary>Réponses de l'IA trop longues, coupées à la fin d'une phrase.</summary>
    public int TruncatedReplies { get; private set; }

    /// <summary>Conversations rejetées, par raison.</summary>
    public Dictionary<string, int> Rejected { get; } = new();

    /// <summary>Messages écartés dès la lecture (supprimés, refusés par les relecteurs), par raison.</summary>
    public Dictionary<string, int> IgnoredMessages { get; } = new();

    /// <param name="lang">Langue gardée ("fr", "en"...).</param>
    /// <param name="maxLength">Longueur max d'un message : MiniChat ne voit que 128 caractères à la fois.</param>
    /// <param name="maxSentences">Nombre max de phrases gardées dans une réponse de l'IA.</param>
    public OasstConverter(string lang, int maxLength, int maxSentences)
    {
        _lang = lang;
        _maxLength = maxLength;
        _maxSentences = maxSentences;
    }

    /// <returns>Le texte au format Toi:/IA:.</returns>
    public string Convert(IEnumerable<string> inputPaths)
    {
        var messages = inputPaths.SelectMany(ReadMessages).ToList();
        var replies = messages.Where(m => m.ParentId is not null).ToLookup(m => m.ParentId!);

        var output = new StringBuilder();

        foreach (var root in messages.Where(m => m.ParentId is null))
        {
            var (lines, stopReason) = FollowBestBranch(root, replies);

            if (lines.Count < 2)   // il faut au moins une question et une réponse
            {
                Count(Rejected, stopReason ?? NoReply);
                continue;
            }

            // Une ligne vide entre deux conversations : MiniChat y met un <|doc|> (une nouvelle conversation commence).
            if (output.Length > 0) output.Append('\n');
            foreach (var line in lines) output.Append(line).Append('\n');
            Kept++;
            if (stopReason is not null) Shortened++;
        }

        return output.ToString();
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
            var (text, rejection) = Clean(current);
            if (text is null)
            {
                stopReason = rejection;   // message inutilisable : la conversation s'arrête là
                break;
            }

            string prefix = current.Role == "prompter" ? "Toi: " : "IA: ";
            lines.Add(prefix + text);

            current = replies[current.Id].OrderBy(m => m.Rank).FirstOrDefault();
        }

        // Une conversation doit finir par une réponse de l'IA.
        if (lines.Count > 0 && lines[^1].StartsWith("Toi: "))
            lines.RemoveAt(lines.Count - 1);

        return (lines, stopReason);
    }

    /// <summary>Texte prêt pour MiniChat, ou null avec la raison du rejet.</summary>
    private (string? Text, string? Rejection) Clean(Message message)
    {
        // Le code informatique n'a aucun sens pour un si petit modèle.
        if (message.Text.Contains("```")) return (null, HasCode);

        // Une réplique = une ligne : MiniChat s'arrête au premier retour à la ligne.
        string text = TextCleaner.SingleLine(TextCleaner.Normalize(message.Text));

        if (text.Length == 0) return (null, Empty);

        // Caractère exotique (emoji, chinois...) : on préfère ignorer le message que le déformer.
        if (!text.All(TextCleaner.IsAllowed)) return (null, ExoticChars);

        // Réponses courtes : un si petit modèle perd le fil sur une longue réponse.
        // (Les questions ne sont jamais coupées : ça changerait leur sens.)
        if (message.Role != "prompter")
        {
            text = TextCleaner.FirstSentences(text, _maxSentences, out bool limited);
            if (limited) LimitedReplies++;
        }

        if (text.Length <= _maxLength) return (text, null);

        // Trop long. Une question coupée changerait de sens : on la rejette.
        // Une réponse, elle, reste valable si on s'arrête à la fin d'une phrase.
        if (message.Role == "prompter") return (null, QuestionTooLong);

        string? cut = TextCleaner.CutAtSentenceEnd(text, _maxLength);
        if (cut is null) return (null, NoSentenceEnd);

        TruncatedReplies++;
        return (cut, null);
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

    private static void Count(Dictionary<string, int> counters, string reason) =>
        counters[reason] = counters.GetValueOrDefault(reason) + 1;

    private static string? GetString(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static bool? GetBool(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : null;
}
