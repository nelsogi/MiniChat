using System.Globalization;
using System.Text;
using DataBuilder;

// Prépare les fichiers de données de MiniChat.
//   oasst      : OpenAssistant (*.messages.jsonl[.gz]) -> dialogues au format Toi:/IA: (dialogues.txt)
//   gutenberg  : livres du Projet Gutenberg (.txt)      -> texte propre pour le pré-entraînement (texte.txt)
//   parquet    : une colonne de texte de fichiers .parquet (Wikipedia, FineWeb...) -> idem

Console.OutputEncoding = Encoding.UTF8;

const string Usage = """
    Utilisation :
      dotnet run --project DataBuilder -- oasst     <sortie.txt> <messages.jsonl[.gz]>... [--lang fr] [--max 300] [--sentences 2]
      dotnet run --project DataBuilder -- gutenberg <sortie.txt> <livre.txt>...
      dotnet run --project DataBuilder -- parquet   <sortie.txt> <fichier.parquet>... [--column text] [--max-mo 50]

    Exemples :
      dotnet run --project DataBuilder -- oasst     Data/dialogues.txt 2023-04-12_oasst_ready.messages.jsonl.gz
      dotnet run --project DataBuilder -- gutenberg Data/texte.txt pg17489.txt pg135.txt
      dotnet run --project DataBuilder -- parquet   Data/texte.txt train-00000-of-00002.parquet --max-mo 50
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

        var converter = new OasstConverter(lang, maxLength, maxSentences);
        output = converter.Convert(inputs);
        PrintOasstReport(converter, lang, maxLength, maxSentences);
        break;

    case "gutenberg":
        var books = new StringBuilder();
        foreach (string input in inputs)
        {
            string cleaned = GutenbergCleaner.Clean(TextCleaner.ReadFile(input), Path.GetFileName(input));
            Console.WriteLine($"  {Path.GetFileName(input)} : {cleaned.Length:N0} caractères.");
            books.Append(cleaned);
        }
        output = books.ToString();
        break;

    case "parquet":
        string column = options.GetValueOrDefault("--column", "text");
        long? maxBytes = options.TryGetValue("--max-mo", out var maxMo)
            ? (long)(double.Parse(maxMo, CultureInfo.InvariantCulture) * 1024 * 1024)
            : null;

        var parquet = new ParquetConverter(column, maxBytes);
        try
        {
            output = await parquet.ConvertAsync(inputs);
        }
        catch (InvalidDataException e)
        {
            Console.WriteLine(e.Message);
            return 1;
        }
        Console.WriteLine($"Documents gardés : {parquet.Kept:N0}"
            + (parquet.ReachedMaxSize ? $" (arrêt à {maxMo} Mo)" : "")
            + $", vides après nettoyage : {parquet.Empty:N0}.");
        break;

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

// Bilan de la conversion : ce qui est gardé, ce qui est rejeté et pourquoi (raisons les plus fréquentes d'abord).
static void PrintOasstReport(OasstConverter converter, string lang, int maxLength, int maxSentences)
{
    int rejected = converter.Rejected.Values.Sum();

    Console.WriteLine($"Langue '{lang}', messages <= {maxLength} caractères, réponses <= {maxSentences} phrases.");
    Console.WriteLine($"Conversations gardées  : {converter.Kept:N0}");
    Console.WriteLine($"  dont arrêtées avant la fin : {converter.Shortened:N0}");
    Console.WriteLine($"  réponses limitées à {maxSentences} phrases : {converter.LimitedReplies:N0}");
    Console.WriteLine($"  réponses coupées en fin de phrase : {converter.TruncatedReplies:N0}");
    Console.WriteLine($"Conversations rejetées : {rejected:N0}");
    foreach (var (reason, count) in converter.Rejected.OrderByDescending(r => r.Value))
        Console.WriteLine($"  {count,6:N0}  {reason}");

    if (converter.IgnoredMessages.Count > 0)
    {
        Console.WriteLine("Messages ignorés à la lecture :");
        foreach (var (reason, count) in converter.IgnoredMessages.OrderByDescending(r => r.Value))
            Console.WriteLine($"  {count,6:N0}  {reason}");
    }
}
