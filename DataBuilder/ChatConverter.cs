using System.IO.Compression;
using System.Text;
using System.Text.Json;
using Parquet;
using Parquet.Serialization;

namespace DataBuilder;

/// <summary>
/// Convertit des conversations en JSON au format de MiniChat (voir DialogueConverter).
/// Fichiers .jsonl (une conversation par ligne), .jsonl.gz, .json (un tableau de conversations)
/// ou .parquet (une conversation par ligne, comme beaucoup de jeux de données de Hugging Face).
///
/// Formats reconnus, les plus courants sur Hugging Face :
///     {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
///     {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}   (ShareGPT)
///     {"instruction": "...", "input": "...", "output": "..."}                                   (Alpaca)
/// Les messages "system" sont ignorés. Une conversation s'arrête au premier message inutilisable.
/// </summary>
public sealed class ChatConverter : DialogueConverter
{
    private const string NotAlternating = "deux messages de suite du même côté";
    private const string UnknownRole = "rôle inconnu";
    private const string UnknownFormat = "format inconnu (ni messages, ni conversations, ni instruction/output)";
    private const string InvalidJson = "JSON illisible";

    private static readonly string[] ListFields = ["messages", "conversations", "conversation", "dialogue"];
    private static readonly string[] RoleFields = ["role", "from", "speaker"];
    private static readonly string[] TextFields = ["content", "value", "text"];
    private static readonly HashSet<string> UserRoles = new(StringComparer.OrdinalIgnoreCase) { "user", "human", "prompter", "utilisateur" };
    private static readonly HashSet<string> AssistantRoles = new(StringComparer.OrdinalIgnoreCase) { "assistant", "gpt", "bot", "model", "chatgpt", "ia" };
    private static readonly HashSet<string> SystemRoles = new(StringComparer.OrdinalIgnoreCase) { "system", "tool" };

    private readonly string? _field;
    private readonly List<(string Field, string[] Values)> _skip;

    /// <summary>Champs du premier enregistrement au format inconnu, pour aider à choisir --field.</summary>
    public string? UnknownFormatFields { get; private set; }

    /// <param name="field">Champ qui contient la liste des messages (null = chercher messages, conversations...).</param>
    /// <param name="skip">
    /// Conversations à écarter selon un champ, null = aucune. Exemple : "source=Logic Inference,Evol Instruct;style=human"
    /// écarte celles dont le champ source contient "Logic Inference" ou "Evol Instruct", et celles dont style contient "human".
    /// </param>
    /// <param name="maxLength">Longueur max d'un message.</param>
    /// <param name="maxSentences">Nombre max de phrases gardées dans une réponse de l'IA.</param>
    public ChatConverter(string? field, string? skip, int maxLength, int maxSentences) : base(maxLength, maxSentences)
    {
        _field = field;
        _skip = ParseSkip(skip);
    }

    private static List<(string Field, string[] Values)> ParseSkip(string? skip)
    {
        var rules = new List<(string, string[])>();
        foreach (string rule in (skip ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int equals = rule.IndexOf('=');
            if (equals <= 0)
                throw new InvalidDataException($"--skip : « {rule} » n'est pas de la forme champ=valeur1,valeur2.");
            string[] values = rule[(equals + 1)..].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            rules.Add((rule[..equals].Trim(), values));
        }
        return rules;
    }

    /// <summary>La règle de --skip qui écarte cet enregistrement ("source contient Logic Inference"), null si aucune.</summary>
    private string? SkipReason(JsonElement record)
    {
        foreach (var (field, values) in _skip)
        {
            if (GetString(record, field) is not { } value) continue;
            foreach (string skipped in values)
            {
                if (value.Contains(skipped, StringComparison.OrdinalIgnoreCase))
                    return $"--skip : {field} contient « {skipped} »";
            }
        }
        return null;
    }

    public override string Convert(IEnumerable<string> inputPaths)
    {
        foreach (string path in inputPaths)
        {
            int keptBefore = Kept;
            foreach (var record in ReadRecords(path))
                AddRecord(record);
            Console.WriteLine($"  {Path.GetFileName(path)} : {Kept - keptBefore:N0} conversations.");
        }
        return Output;
    }

    private void AddRecord(JsonElement record)
    {
        if (record.ValueKind != JsonValueKind.Object)
        {
            Count(IgnoredMessages, UnknownFormat);
            return;
        }

        if (SkipReason(record) is { } reason)
        {
            Count(IgnoredMessages, reason);
            return;
        }

        if (FindMessages(record) is { } messages)
        {
            AddConversation(messages);
            return;
        }

        // Alpaca : une seule question (instruction + input éventuel) et sa réponse.
        if (GetString(record, "instruction") is { } instruction && GetString(record, "output") is { } output)
        {
            string input = GetString(record, "input") ?? "";
            string question = input.Trim().Length > 0 ? instruction + "\n" + input : instruction;
            AddConversation([(true, question), (false, output)]);
            return;
        }

        Count(IgnoredMessages, UnknownFormat);
        UnknownFormatFields ??= string.Join(", ", record.EnumerateObject().Select(p => p.Name));
    }

    /// <summary>La liste des messages (côté utilisateur ?, texte), null si l'enregistrement n'en a pas.</summary>
    private List<(bool? FromUser, string Text)>? FindMessages(JsonElement record)
    {
        JsonElement list = default;
        bool found = _field is not null
            ? record.TryGetProperty(_field, out list)
            : ListFields.Any(name => record.TryGetProperty(name, out list));
        if (!found || list.ValueKind != JsonValueKind.Array) return null;

        var messages = new List<(bool? FromUser, string Text)>();
        foreach (var message in list.EnumerateArray())
        {
            string? role = RoleFields.Select(name => GetString(message, name)).FirstOrDefault(r => r is not null);
            if (role is not null && SystemRoles.Contains(role)) continue;

            bool? fromUser = role is null ? null
                : UserRoles.Contains(role) ? true
                : AssistantRoles.Contains(role) ? false
                : null;
            string text = TextFields.Select(name => GetString(message, name)).FirstOrDefault(t => t is not null) ?? "";
            messages.Add((fromUser, text));
        }
        return messages;
    }

    private void AddConversation(List<(bool? FromUser, string Text)> messages)
    {
        var lines = new List<string>();
        string? stopReason = null;

        foreach (var (fromUser, rawText) in messages)
        {
            if (fromUser is null)
            {
                stopReason = UnknownRole;
                break;
            }

            // Certaines conversations commencent par un message d'accueil de l'assistant : on le saute.
            if (lines.Count == 0 && fromUser == false) continue;

            if (fromUser != (lines.Count % 2 == 0))
            {
                stopReason = NotAlternating;
                break;
            }

            var (text, rejection) = Clean(rawText, fromUser.Value);
            if (text is null)
            {
                stopReason = rejection;   // message inutilisable : la conversation s'arrête là
                break;
            }

            lines.Add((fromUser.Value ? "Toi: " : "IA: ") + text);
        }

        Add(lines, stopReason);
    }

    /// <summary>
    /// Les conversations du fichier : une par ligne (.jsonl), les éléments d'un tableau (.json),
    /// ou les lignes d'un .parquet.
    /// </summary>
    private IEnumerable<JsonElement> ReadRecords(string path)
    {
        if (path.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase))
            return ReadParquet(path);
        return ReadJson(path);
    }

    /// <summary>
    /// Chaque ligne du .parquet, convertie en JSON : les listes de messages imbriquées deviennent des
    /// tableaux d'objets, exactement comme dans un .jsonl. Un groupe de lignes à la fois, pour la mémoire.
    /// </summary>
    private static IEnumerable<JsonElement> ReadParquet(string path)
    {
        var reader = ParquetReader.CreateAsync(path).GetAwaiter().GetResult();
        int rowGroups = reader.RowGroupCount;
        reader.DisposeAsync().AsTask().GetAwaiter().GetResult();

        for (int i = 0; i < rowGroups; i++)
        {
            List<Dictionary<string, object>> rows;
            using (var file = File.OpenRead(path))
                rows = ParquetSerializer.DeserializeUntypedAsync(file, null, i).GetAwaiter().GetResult().Data.ToList();

            foreach (var row in rows)
                yield return JsonSerializer.SerializeToElement(row);
        }
    }

    private IEnumerable<JsonElement> ReadJson(string path)
    {
        using var file = File.OpenRead(path);
        using Stream stream = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase)
            ? new GZipStream(file, CompressionMode.Decompress)
            : file;

        string name = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? path[..^3] : path;
        if (name.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            using var document = JsonDocument.Parse(stream);
            if (document.RootElement.ValueKind != JsonValueKind.Array)
                throw new InvalidDataException($"{Path.GetFileName(path)} : un fichier .json doit contenir un tableau de conversations.");
            foreach (var record in document.RootElement.EnumerateArray())
                yield return record;
            yield break;
        }

        using var reader = new StreamReader(stream, Encoding.UTF8);
        while (reader.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            JsonDocument document;
            try
            {
                document = JsonDocument.Parse(line);
            }
            catch (JsonException)
            {
                Count(IgnoredMessages, InvalidJson);
                continue;
            }
            using (document)
                yield return document.RootElement;
        }
    }

    private static string? GetString(JsonElement e, string name) =>
        e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString()
            : null;
}
