using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace DataBuilder;

/// <summary>
/// Fabrique des dialogues de calcul au format de MiniChat (Toi:/IA:, une ligne vide entre deux) : additions,
/// soustractions et multiplications de nombres de 0 à 100, avec des formulations variées (« Combien font 7 + 5 ? »,
/// « Quelle est la somme de 7 et 5 ? », un petit problème avec des billes...). Graine fixe : mêmes dialogues à chaque fois.
///
/// Aucune question n'est une question de eval.json (même texte, sans tenir compte des majuscules, des accents ni de la
/// ponctuation), et aucun calcul n'est un calcul de eval.json (7 + 5 n'est jamais demandé si eval.json le demande, ni
/// 5 + 7, ni 3 fois 4 / 4 fois 3) : /eval mesure si le modèle sait calculer, pas s'il a appris la réponse par cœur.
/// Les soustractions ne donnent jamais de nombre négatif. Une question n'est posée qu'une fois.
/// </summary>
public sealed partial class CalculGenerator
{
    public enum Operation { Addition, Subtraction, Multiplication }

    /// <summary>Un calcul : a op b. Pour l'addition et la multiplication, a &lt;= b (5 + 7 et 7 + 5 sont le même).</summary>
    public readonly record struct Problem(Operation Op, int A, int B)
    {
        public int Result => Op switch
        {
            Operation.Addition => A + B,
            Operation.Subtraction => A - B,
            _ => A * B,
        };

        public static Problem Of(Operation op, int a, int b) =>
            op == Operation.Subtraction || a <= b ? new(op, a, b) : new(op, b, a);
    }

    /// <summary>Une formulation : la question et ses réponses possibles, avec {a}, {b} et {r} (le résultat). Min : plus petit nombre (2 pour un petit problème : ni « 0 crayons » ni « 1 livres »).</summary>
    private sealed record Template(string Question, string[] Replies, int Min = 0);

    public const int MaxNumber = 100;

    private static readonly string[] CommonReplies = ["Cela fait {r}.", "Ça fait {r}.", "Le résultat est {r}.", "La réponse est {r}.", "{r}."];

    private static readonly Dictionary<Operation, Template[]> Templates = new()
    {
        [Operation.Addition] =
        [
            new("Combien font {a} + {b} ?", ["{a} + {b} = {r}.", .. CommonReplies]),
            new("Combien fait {a} plus {b} ?", ["{a} plus {b} font {r}.", .. CommonReplies]),
            new("Calcule {a} + {b}.", ["{a} + {b} = {r}.", .. CommonReplies]),
            new("Que vaut {a} + {b} ?", ["{a} + {b} vaut {r}.", .. CommonReplies]),
            new("{a} + {b} = ?", ["{a} + {b} = {r}.", "{r}."]),
            new("Quel est le résultat de {a} + {b} ?", ["Le résultat est {r}.", "{a} + {b} = {r}."]),
            new("Peux-tu calculer {a} plus {b} ?", ["Bien sûr : {a} plus {b} font {r}.", "Oui, cela fait {r}."]),
            new("Quelle est la somme de {a} et {b} ?", ["La somme de {a} et {b} est {r}.", "Elle vaut {r}."]),
            new("Additionne {a} et {b}.", ["{a} + {b} = {r}.", "Cela fait {r}."]),
            new("J'ajoute {b} à {a}, combien j'obtiens ?", ["Tu obtiens {r}.", "{a} + {b} = {r}."]),
            new("J'ai {a} billes et on m'en donne {b}. Combien en ai-je maintenant ?", ["Tu as maintenant {r} billes.", "{a} + {b} = {r}, donc {r} billes."], Min: 2),
            new("Dans une classe, il y a {a} filles et {b} garçons. Combien y a-t-il d'élèves ?", ["Il y a {r} élèves.", "{a} + {b} = {r} élèves."], Min: 2),
        ],
        [Operation.Subtraction] =
        [
            new("Combien font {a} - {b} ?", ["{a} - {b} = {r}.", .. CommonReplies]),
            new("Combien fait {a} moins {b} ?", ["{a} moins {b} font {r}.", .. CommonReplies]),
            new("Calcule {a} - {b}.", ["{a} - {b} = {r}.", .. CommonReplies]),
            new("Que vaut {a} moins {b} ?", ["{a} moins {b} vaut {r}.", .. CommonReplies]),
            new("{a} - {b} = ?", ["{a} - {b} = {r}.", "{r}."]),
            new("Quel est le résultat de {a} - {b} ?", ["Le résultat est {r}.", "{a} - {b} = {r}."]),
            new("Quelle est la différence entre {a} et {b} ?", ["La différence entre {a} et {b} est {r}.", "{a} - {b} = {r}."]),
            new("Si j'enlève {b} à {a}, combien reste-t-il ?", ["Il reste {r}.", "{a} - {b} = {r}."]),
            new("Soustrais {b} de {a}.", ["{a} - {b} = {r}.", "Cela fait {r}."]),
            new("J'avais {a} euros et j'en dépense {b}. Combien m'en reste-t-il ?", ["Il te reste {r} euros.", "{a} - {b} = {r}, donc {r} euros."], Min: 2),
            new("Un bus transporte {a} personnes. {b} descendent. Combien en reste-t-il dans le bus ?", ["Il en reste {r}.", "{a} - {b} = {r} personnes."], Min: 2),
        ],
        [Operation.Multiplication] =
        [
            new("Combien font {a} fois {b} ?", ["{a} fois {b} font {r}.", .. CommonReplies]),
            new("Combien font {a} x {b} ?", ["{a} x {b} = {r}.", .. CommonReplies]),
            new("Combien fait {a} multiplié par {b} ?", ["{a} multiplié par {b} font {r}.", .. CommonReplies]),
            new("Calcule {a} x {b}.", ["{a} x {b} = {r}.", .. CommonReplies]),
            new("Que vaut {a} fois {b} ?", ["{a} fois {b} vaut {r}.", .. CommonReplies]),
            new("{a} x {b} = ?", ["{a} x {b} = {r}.", "{r}."]),
            new("Quel est le produit de {a} et {b} ?", ["Le produit de {a} et {b} est {r}.", "Il vaut {r}."]),
            new("Multiplie {a} par {b}.", ["{a} x {b} = {r}.", "Cela fait {r}."]),
            new("Il y a {a} boîtes de {b} crayons. Combien de crayons en tout ?", ["Il y a {r} crayons en tout.", "{a} x {b} = {r} crayons."], Min: 2),
            new("Un livre coûte {b} euros. Combien coûtent {a} livres ?", ["{a} livres coûtent {r} euros.", "{a} x {b} = {r}, donc {r} euros."], Min: 2),
        ],
    };

    private readonly HashSet<string> _excludedQuestions;
    private readonly HashSet<Problem> _excludedProblems;

    /// <summary>Questions de eval.json lues.</summary>
    public int EvalQuestions { get; }

    /// <summary>Calculs de eval.json reconnus (jamais demandés).</summary>
    public int ExcludedProblems => _excludedProblems.Count;

    /// <summary>Dialogues fabriqués par opération.</summary>
    public Dictionary<Operation, int> Counts { get; } = new();

    /// <param name="evalQuestions">Les questions de eval.json : ni elles, ni leurs calculs ne sont repris.</param>
    public CalculGenerator(IEnumerable<string> evalQuestions)
    {
        var questions = evalQuestions.ToList();
        EvalQuestions = questions.Count;
        _excludedQuestions = questions.Select(Simplify).ToHashSet();
        _excludedProblems = questions.SelectMany(ProblemsIn).ToHashSet();
    }

    /// <summary>Les questions d'un fichier eval.json ({"questions": [{"question": "..."}]}).</summary>
    public static IEnumerable<string> ReadEvalQuestions(string path)
    {
        using var json = JsonDocument.Parse(File.ReadAllText(path));
        foreach (var question in json.RootElement.GetProperty("questions").EnumerateArray())
            if (question.TryGetProperty("question", out var text) && text.GetString() is { } value)
                yield return value;
    }

    /// <summary>count dialogues d'un tour (question, réponse), dans l'ordre du tirage.</summary>
    public List<(string Question, string Reply, Problem Problem)> Generate(int count, int seed)
    {
        var random = new Random(seed);
        var questions = new HashSet<string>();
        var dialogues = new List<(string, string, Problem)>(count);
        var operations = Enum.GetValues<Operation>();

        // 3 opérations x ~5 000 calculs x ~11 formulations : bien plus de questions que demandé, la boucle finit vite.
        while (dialogues.Count < count)
        {
            var op = operations[dialogues.Count % operations.Length];   // autant de chaque opération
            var templates = Templates[op];
            var template = templates[random.Next(templates.Length)];
            string reply = template.Replies[random.Next(template.Replies.Length)];
            int a = random.Next(template.Min, MaxNumber + 1), b = random.Next(template.Min, MaxNumber + 1);
            if (op == Operation.Subtraction && a < b) (a, b) = (b, a);   // pas de négatif

            var problem = Problem.Of(op, a, b);
            string question = Fill(template.Question, a, b, problem.Result);
            if (_excludedProblems.Contains(problem) || _excludedQuestions.Contains(Simplify(question)) || !questions.Add(question))
                continue;

            dialogues.Add((question, Fill(reply, a, b, problem.Result), problem));
            Counts[op] = Counts.GetValueOrDefault(op) + 1;
        }
        return dialogues;
    }

    /// <summary>Les dialogues au format de dialogues.txt.</summary>
    public static string Format(IEnumerable<(string Question, string Reply, Problem Problem)> dialogues)
    {
        var output = new StringBuilder();
        foreach (var (question, reply, _) in dialogues)
        {
            if (output.Length > 0) output.Append('\n');
            output.Append("Toi: ").Append(question).Append('\n').Append("IA: ").Append(reply).Append('\n');
        }
        return output.ToString();
    }

    /// <summary>
    /// Les calculs d'une question : « 7 + 5 », « 10 - 3 », « 3 fois 4 », « 6 x 7 », « 9 plus 8 »... Un calcul
    /// qui n'a pas de sens pour CalculGenerator (division, nombre au-delà de 100) est ignoré : il n'est jamais généré.
    /// </summary>
    public static IEnumerable<Problem> ProblemsIn(string question)
    {
        foreach (Match match in Calculation().Matches(question))
        {
            int a = int.Parse(match.Groups["a"].Value, CultureInfo.InvariantCulture);
            int b = int.Parse(match.Groups["b"].Value, CultureInfo.InvariantCulture);
            Operation? op = match.Groups["op"].Value.ToLowerInvariant() switch
            {
                "+" or "plus" => Operation.Addition,
                "-" or "moins" => Operation.Subtraction,
                "x" or "*" or "×" or "fois" or "multiplié par" or "multiplie par" => Operation.Multiplication,
                _ => null,
            };
            if (op is { } operation)
                yield return Problem.Of(operation, a, b);
        }
        // « le double de 16 » : 2 fois 16.
        foreach (Match match in Double().Matches(question))
            yield return Problem.Of(Operation.Multiplication, 2, int.Parse(match.Groups["n"].Value, CultureInfo.InvariantCulture));
    }

    /// <summary>
    /// Minuscules, sans accents, et tout ce qui n'est ni lettre ni chiffre devient une espace :
    /// « Combien font 7 + 5 ? » et « combien font 7 5 » sont la même question (comme EvalTest.Normalize de MiniChat).
    /// </summary>
    public static string Simplify(string text)
    {
        var simple = new StringBuilder(text.Length);
        foreach (char c in text.ToLowerInvariant().Replace("œ", "oe").Replace("æ", "ae").Normalize(NormalizationForm.FormKD))
        {
            if (CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.NonSpacingMark) continue;
            simple.Append(char.IsLetterOrDigit(c) ? c : ' ');
        }
        return string.Join(' ', simple.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries));
    }

    private static string Fill(string template, int a, int b, int result) => template
        .Replace("{a}", a.ToString(CultureInfo.InvariantCulture))
        .Replace("{b}", b.ToString(CultureInfo.InvariantCulture))
        .Replace("{r}", result.ToString(CultureInfo.InvariantCulture));

    [GeneratedRegex(@"(?<a>\d+)\s*(?<op>\+|-|x|\*|×|fois|plus|moins|multiplié par|multiplie par)\s*(?<b>\d+)", RegexOptions.IgnoreCase)]
    private static partial Regex Calculation();

    [GeneratedRegex(@"double de (?<n>\d+)", RegexOptions.IgnoreCase)]
    private static partial Regex Double();
}
