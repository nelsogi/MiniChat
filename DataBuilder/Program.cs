using System.Globalization;
using System.Text;
using DataBuilder;

// Prépare les fichiers de données de MiniChat.
//   oasst      : OpenAssistant (*.messages.jsonl[.gz]) -> dialogues au format Toi:/IA: (dialogues.txt)
//   chat       : conversations en JSON (messages, ShareGPT, Alpaca)   -> idem
//   gutenberg  : livres du Projet Gutenberg (.txt)      -> texte propre pour le pré-entraînement (texte.txt)
//   parquet    : une colonne de texte de fichiers .parquet (Wikipedia, FineWeb...) -> idem
//   calcul     : dialogues de calcul fabriqués (additions, soustractions, multiplications) -> dialogues
//   sans-eval  : retire d'un dialogues.txt les conversations qui contiennent une question de eval.json

Console.OutputEncoding = Encoding.UTF8;

const string Usage = """
    Utilisation :
      dotnet run --project DataBuilder -- oasst     <sortie.txt> <messages.jsonl[.gz]>... [--lang fr] [--max 300] [--sentences 2] [--min-turns 2]
      dotnet run --project DataBuilder -- chat      <sortie.txt> <fichier.jsonl[.gz]|.json|.parquet>... [--field messages] [--skip "source=a,b;style=c"] [--max 300] [--sentences 2] [--append oui]
      dotnet run --project DataBuilder -- gutenberg <sortie.txt> <livre.txt>...
      dotnet run --project DataBuilder -- parquet   <sortie.txt> <fichier.parquet>... [--column text] [--max-mo 50] [--append oui] [--exclude a.parquet,b.parquet]
      dotnet run --project DataBuilder -- calcul    <sortie.txt> <eval.json> [--count 20000] [--seed 1] [--append oui]
      dotnet run --project DataBuilder -- sans-eval <dialogues.txt> <eval.json>

    Exemples :
      dotnet run --project DataBuilder -- oasst     Data/dialogues.txt 2023-11-05_oasst2_ready.messages.jsonl.gz
      dotnet run --project DataBuilder -- chat      Data/dialogues.txt conversations-fr.jsonl --append oui
      dotnet run --project DataBuilder -- gutenberg Data/texte.txt pg17489.txt pg135.txt
      dotnet run --project DataBuilder -- parquet   Data/texte.txt train-00000-of-00002.parquet --max-mo 50
      dotnet run --project DataBuilder -- calcul    Data/dialogues.txt Data/eval.json --append oui
    """;

// Sépare les options (--lang fr) des chemins de fichiers.
var options = new Dictionary<string, string>();
var paths = new List<string>();
for (int i = 0; i < args.Length; i++)
{
    if (args[i].StartsWith("--") && i + 1 < args.Length) options[args[i]] = args[++i];
    else paths.Add(args[i]);
}

if (paths.Count < 3)
{
    Console.WriteLine(Usage);
    return 1;
}

string command = paths[0];
string outputPath = paths[1];
var inputs = paths.Skip(2).ToList();

var missing = inputs.Where(p => !File.Exists(p)).ToList();
if (missing.Count > 0)
{
    Console.WriteLine($"Fichier introuvable : {string.Join(", ", missing)}");
    return 1;
}

string output;
switch (command)
{
    case "oasst":
        string lang = options.GetValueOrDefault("--lang", "fr");
        int maxLength = int.Parse(options.GetValueOrDefault("--max", "300"));
        int maxSentences = int.Parse(options.GetValueOrDefault("--sentences", "2"));
        int minTurns = int.Parse(options.GetValueOrDefault("--min-turns", "2"));

        var converter = new OasstConverter(lang, maxLength, maxSentences, minTurns);
        output = converter.Convert(inputs);
        Console.WriteLine($"Langue '{lang}'.");
        PrintDialogueReport(converter);
        PrintTurns(converter);
        break;

    case "calcul":
        int count = int.Parse(options.GetValueOrDefault("--count", "20000"));
        int seed = int.Parse(options.GetValueOrDefault("--seed", "1"));
        var generator = new CalculGenerator(inputs.SelectMany(CalculGenerator.ReadEvalQuestions));
        output = CalculGenerator.Format(generator.Generate(count, seed));
        Console.WriteLine($"{generator.EvalQuestions} questions d'évaluation lues : ni elles, ni leurs " +
                          $"{generator.ExcludedProblems} calculs ne sont repris.");
        Console.WriteLine($"Dialogues : {count:N0} (graine {seed}) : " + string.Join(", ",
            generator.Counts.Select(c => $"{c.Value:N0} {c.Key switch {
                CalculGenerator.Operation.Addition => "additions",
                CalculGenerator.Operation.Subtraction => "soustractions",
                _ => "multiplications" }}")) + ".");
        Console.WriteLine("Par nombre de tours : " + string.Join(", ",
            generator.TurnCounts.Select(t => $"{t.Value:N0} de {t.Key} tour{(t.Key > 1 ? "s" : "")}")) + ".");
        output = AppendTo(outputPath, output, options);
        break;

    case "sans-eval":
        if (!File.Exists(outputPath))
        {
            Console.WriteLine($"Fichier introuvable : {outputPath}");
            return 1;
        }
        var filter = new EvalFilter(inputs.SelectMany(CalculGenerator.ReadEvalQuestions));
        output = filter.Filter(TextCleaner.ReadFile(outputPath));
        Console.WriteLine($"Conversations gardées : {filter.Kept:N0}, retirées : {filter.Removed:N0} " +
                          $"({filter.Found.Count} questions d'évaluation trouvées).");
        foreach (var (question, found) in filter.Found.OrderByDescending(f => f.Value))
            Console.WriteLine($"  {found,6:N0}  {question}");
        break;

    case "chat":
        ChatConverter chat;
        try
        {
            chat = new ChatConverter(
                options.GetValueOrDefault("--field"),
                options.GetValueOrDefault("--skip"),
                int.Parse(options.GetValueOrDefault("--max", "300")),
                int.Parse(options.GetValueOrDefault("--sentences", "2")));
            output = chat.Convert(inputs);
        }
        catch (Exception e) when (e is InvalidDataException or System.Text.Json.JsonException)
        {
            Console.WriteLine(e.Message);
            return 1;
        }
        PrintDialogueReport(chat);
        if (chat.Kept == 0)
        {
            // Rien de gardé : souvent un format que ChatConverter ne connaît pas. On n'écrase pas la sortie.
            if (chat.UnknownFormatFields is not null)
                Console.WriteLine($"Champs trouvés : {chat.UnknownFormatFields}. Indique celui des messages avec --field.");
            Console.WriteLine($"Aucune conversation gardée : {outputPath} n'est pas écrit.");
            return 1;
        }

        output = AppendTo(outputPath, output, options);
        break;

    case "gutenberg":
        var books = new StringBuilder();
        foreach (string input in inputs)
        {
            string cleaned = GutenbergCleaner.Clean(TextCleaner.ReadFile(input), Path.GetFileName(input));
            Console.WriteLine($"  {Path.GetFileName(input)} : {cleaned.Length:N0} caractères.");
            books.Append(TextCleaner.DocToken).Append(cleaned);   // un livre = un document
        }
        output = books.ToString();
        break;

    case "parquet":
        string column = options.GetValueOrDefault("--column", "text");
        long? maxBytes = options.TryGetValue("--max-mo", out var maxMo)
            ? (long)(double.Parse(maxMo, CultureInfo.InvariantCulture) * 1024 * 1024)
            : null;

        // --append oui : à la fin du fichier (texte.txt déjà fait), sans reprendre ses documents.
        // --exclude : fichiers .parquet déjà utilisés, dont les documents ne sont pas repris non plus.
        bool append = options.GetValueOrDefault("--append") == "oui";
        var exclude = options.GetValueOrDefault("--exclude", "")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
        if (exclude.FirstOrDefault(p => !File.Exists(p)) is { } missingExclude)
        {
            Console.WriteLine($"Fichier introuvable : {missingExclude}");
            return 1;
        }

        // Écrit lui-même le fichier au fur et à mesure (la sortie peut dépasser la taille max d'une chaîne).
        var parquet = new ParquetConverter(column, maxBytes);
        try
        {
            await parquet.ConvertAsync(inputs, outputPath, append, exclude);
        }
        catch (InvalidDataException e)
        {
            Console.WriteLine(e.Message);
            return 1;
        }
        Console.WriteLine($"Documents gardés : {parquet.Kept:N0}"
            + (parquet.ReachedMaxSize ? $" (arrêt à {maxMo} Mo)" : "")
            + $", vides après nettoyage : {parquet.Empty:N0}, doublons écartés : {parquet.Duplicates:N0}.");
        if (append)
            Console.WriteLine($"Ajoutés à la suite de {outputPath}.");
        Console.WriteLine($"Écrit : {outputPath} ({parquet.Characters:N0} caractères, {parquet.Vocabulary} caractères différents).");
        return 0;

    default:
        Console.WriteLine($"Commande inconnue : {command}\n\n{Usage}");
        return 1;
}

// UTF-8 sans BOM, fins de ligne "\n" : exactement ce que MiniChat attend.
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
File.WriteAllText(outputPath, output, new UTF8Encoding(false));

int vocabulary = output.Distinct().Count();
Console.WriteLine($"Écrit : {outputPath} ({output.Length:N0} caractères, {vocabulary} caractères différents).");
return 0;

// ---------------------------------------------------------------------------

// --append oui : output à la fin d'un dialogues.txt existant (fait par exemple avec oasst), une ligne vide entre les deux.
static string AppendTo(string outputPath, string output, Dictionary<string, string> options)
{
    if (options.GetValueOrDefault("--append") != "oui" || !File.Exists(outputPath))
        return output;
    string existing = TextCleaner.ReadFile(outputPath).TrimEnd('\n');
    if (existing.Length == 0)
        return output;
    Console.WriteLine($"Ajouté à la suite de {outputPath}.");
    return existing + "\n\n" + output;
}

// Conversations par nombre de tours (question + réponse), avant (une par arbre) et après (chemins jusqu'aux feuilles).
static void PrintTurns(OasstConverter converter)
{
    var turns = converter.TurnsBefore.Keys.Union(converter.TurnsAfter.Keys).Order().ToList();
    Console.WriteLine($"Conversations par nombre de tours : avant (une par arbre, arrêt au premier message inutilisable) " +
                      $"-> après (chemins jusqu'aux feuilles, {converter.MinTurns} tour{(converter.MinTurns > 1 ? "s" : "")} au moins) :");
    foreach (int turn in turns)
        Console.WriteLine($"  {turn,3} tour{(turn > 1 ? "s" : " ")} : {converter.TurnsBefore.GetValueOrDefault(turn),6:N0} -> {converter.TurnsAfter.GetValueOrDefault(turn),6:N0}");
    Console.WriteLine($"  total      : {converter.TurnsBefore.Values.Sum(),6:N0} -> {converter.TurnsAfter.Values.Sum(),6:N0}");
}

// Bilan de la conversion : ce qui est gardé, ce qui est rejeté et pourquoi (raisons les plus fréquentes d'abord).
static void PrintDialogueReport(DialogueConverter converter)
{
    int rejected = converter.Rejected.Values.Sum();
    int maxSentences = converter.MaxSentences;

    Console.WriteLine($"Messages <= {converter.MaxLength} caractères, réponses <= {maxSentences} phrases.");
    Console.WriteLine($"Conversations gardées  : {converter.Kept:N0}");
    Console.WriteLine($"  dont arrêtées avant la fin : {converter.Shortened:N0}");
    Console.WriteLine($"  réponses limitées à {maxSentences} phrases : {converter.LimitedReplies:N0}");
    Console.WriteLine($"  réponses coupées en fin de phrase : {converter.TruncatedReplies:N0}");
    Console.WriteLine($"Conversations rejetées : {rejected:N0}");
    foreach (var (reason, count) in converter.Rejected.OrderByDescending(r => r.Value))
        Console.WriteLine($"  {count,6:N0}  {reason}");

    if (converter.IgnoredMessages.Count > 0)
    {
        Console.WriteLine("Ignorés à la lecture :");
        foreach (var (reason, count) in converter.IgnoredMessages.OrderByDescending(r => r.Value))
            Console.WriteLine($"  {count,6:N0}  {reason}");
    }
}
