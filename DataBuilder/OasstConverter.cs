using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace DataBuilder;

/// <summary>
/// Convertit OpenAssistant (fichier "*.messages.jsonl" ou ".jsonl.gz", un message JSON par ligne)
/// au format de MiniChat (voir DialogueConverter).
///
/// Dans OpenAssistant, les conversations sont des arbres : chaque message peut avoir plusieurs
/// réponses, classées par des humains ("rank", 0 = la meilleure). On garde les conversations complètes :
/// tous les chemins de la question de départ jusqu'aux feuilles. Après une question, on prend la réponse
/// de l'IA la mieux classée (la suivante si elle est inutilisable) ; après une réponse, chaque relance de
/// l'utilisateur ouvre un chemin. Un chemin de moins de MinTurns tours (question + réponse) n'est pas gardé.
/// </summary>
public sealed class OasstConverter : DialogueConverter
{
    private const string TooFewTurns = "moins de {0} tours";

    private sealed record Message(string Id, string? ParentId, string Role, string Text, int Rank);

    /// <summary>Un chemin : ses lignes Toi:/IA:, et pourquoi il s'arrête avant une feuille (null s'il va au bout).</summary>
    private sealed record Branch(List<string> Lines, string? StopReason);

    private readonly string _lang;
    private readonly Dictionary<string, (string? Text, string? Rejection)> _cleaned = new();
    private ILookup<string, Message> _replies = Array.Empty<Message>().ToLookup(m => m.Id);

    /// <summary>Nombre minimum de tours (question + réponse) d'une conversation gardée.</summary>
    public int MinTurns { get; }

    /// <summary>
    /// Conversations par nombre de tours avec l'ancienne extraction (une par arbre, en suivant toujours le
    /// message le mieux classé, arrêt au premier inutilisable), pour comparer.
    /// </summary>
    public SortedDictionary<int, int> TurnsBefore { get; } = new();

    /// <summary>Conversations gardées par nombre de tours.</summary>
    public SortedDictionary<int, int> TurnsAfter { get; } = new();

    /// <param name="lang">Langue gardée ("fr", "en"...).</param>
    /// <param name="maxLength">Longueur max d'un message.</param>
    /// <param name="maxSentences">Nombre max de phrases gardées dans une réponse de l'IA.</param>
    /// <param name="minTurns">Nombre minimum de tours d'une conversation gardée.</param>
    public OasstConverter(string lang, int maxLength, int maxSentences, int minTurns = 2) : base(maxLength, maxSentences)
    {
        _lang = lang;
        MinTurns = minTurns;
    }

    public override string Convert(IEnumerable<string> inputPaths) =>
        Convert(inputPaths.SelectMany(ReadMessages));

    /// <summary>Des messages déjà lus (un ou plusieurs arbres). Rank null = pas classé.</summary>
    public string Convert(IEnumerable<(string Id, string? ParentId, string Role, string Text, int? Rank)> messages) =>
        Convert(messages.Select(m => new Message(m.Id, m.ParentId, m.Role, m.Text, m.Rank ?? int.MaxValue)));

    private string Convert(IEnumerable<Message> messages)
    {
        var all = messages.ToList();
        _replies = all.Where(m => m.ParentId is not null).ToLookup(m => m.ParentId!);
        var before = new HashSet<string>();

        foreach (var root in all.Where(m => m.ParentId is null && m.Role == "prompter"))
        {
            if (BestBranch(root) is { Count: >= 2 } old && before.Add(string.Join('\n', old)))
                Count(TurnsBefore, old.Count / 2);

            var branches = FromQuestion(root, out string? rejection);
            if (branches.Count == 0)
            {
                Add([], rejection);
                continue;
            }
            foreach (var branch in branches)
            {
                int turns = branch.Lines.Count / 2;
                if (turns < MinTurns)
                    Count(Rejected, string.Format(TooFewTurns, MinTurns));
                else if (Add(branch.Lines, branch.StopReason))
                    Count(TurnsAfter, turns);
            }
        }

        return Output;
    }

    /// <summary>
    /// Les chemins qui partent de cette question : la réponse la mieux classée qui est utilisable, suivie de
    /// chaque relance de l'utilisateur. Aucun (et rejection dit pourquoi) si la question ou toutes ses
    /// réponses sont inutilisables.
    /// </summary>
    private List<Branch> FromQuestion(Message question, out string? rejection)
    {
        var (text, questionRejection) = Cleaned(question);
        rejection = questionRejection ?? NoReply;
        if (text is null) return [];

        foreach (var reply in _replies[question.Id].Where(m => m.Role == "assistant").OrderBy(m => m.Rank))
        {
            var (answer, answerRejection) = Cleaned(reply);
            if (answer is null)
            {
                rejection = answerRejection;
                continue;
            }

            var head = new List<string> { "Toi: " + text, "IA: " + answer };
            var branches = new List<Branch>();
            string? stopReason = null;
            foreach (var followUp in _replies[reply.Id].Where(m => m.Role == "prompter").OrderBy(m => m.Rank))
            {
                var tails = FromQuestion(followUp, out string? tailRejection);
                stopReason ??= tails.Count == 0 ? tailRejection : null;
                branches.AddRange(tails.Select(t => new Branch([.. head, .. t.Lines], t.StopReason)));
            }

            // Une feuille, ou des relances toutes inutilisables : la conversation s'arrête à cette réponse.
            if (branches.Count == 0)
                branches.Add(new Branch(head, stopReason));
            rejection = null;
            return branches;
        }
        return [];
    }

    /// <summary>L'ancienne extraction : toujours le message le mieux classé, arrêt au premier inutilisable.</summary>
    private List<string> BestBranch(Message root)
    {
        var lines = new List<string>();
        for (Message? current = root; current is not null; current = _replies[current.Id].OrderBy(m => m.Rank).FirstOrDefault())
        {
            if (Cleaned(current).Text is not { } text) break;
            lines.Add((current.Role == "prompter" ? "Toi: " : "IA: ") + text);
        }
        if (lines.Count > 0 && lines[^1].StartsWith("Toi: "))
            lines.RemoveAt(lines.Count - 1);
        return lines;
    }

    /// <summary>Clean une seule fois par message : un message est sur plusieurs chemins, les compteurs le comptent une fois.</summary>
    private (string? Text, string? Rejection) Cleaned(Message message)
    {
        if (!_cleaned.TryGetValue(message.Id, out var result))
            _cleaned[message.Id] = result = Clean(message.Text, message.Role == "prompter");
        return result;
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

    private static void Count(SortedDictionary<int, int> counters, int turns) =>
        counters[turns] = counters.GetValueOrDefault(turns) + 1;

    private static string? GetString(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static bool? GetBool(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : null;
}
