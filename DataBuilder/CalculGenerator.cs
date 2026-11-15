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
///
/// MultiTurnShare des conversations (30 %) ont 2 ou 3 tours : les questions d'après reprennent le résultat précédent
/// (« Et si on ajoute 4 ? », « Multiplie ça par 2. »), pour que le modèle apprenne à suivre une conversation. Ces
/// calculs-là non plus ne sont jamais des calculs de eval.json.
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

    /// <summary>Un tour : la question, la réponse, et le calcul demandé.</summary>
    public sealed record Turn(string Question, string Reply, Problem Problem);

    /// <summary>Une conversation de 1 à 3 tours.</summary>
    public sealed record Dialogue(IReadOnlyList<Turn> Turns);

    /// <summary>Part des conversations à 2 ou 3 tours : 3 sur 10.</summary>
    public const double MultiTurnShare = 0.3;

    /// <summary>
    /// Une question qui reprend le résultat précédent ({a}) avec un nombre {b} (Fixed : toujours le même, « Double ce
    /// résultat. ») ; {r} = le nouveau résultat.
    /// </summary>
    private sealed record FollowUp(Operation Op, string Question, string[] Replies, int? Fixed = null);

    private static readonly FollowUp[] FollowUps =
    [
        new(Operation.Addition, "Et si on ajoute {b} ?", ["{a} + {b} = {r}.", "Ça fait {r}.", "Alors on obtient {r}."]),
        new(Operation.Addition, "Et si j'en ajoute {b} ?", ["{a} + {b} = {r}.", "Tu en as alors {r}."]),
        new(Operation.Addition, "Ajoute {b} à ce résultat.", ["{a} + {b} = {r}.", "Ça fait {r}."]),
        new(Operation.Addition, "Plus {b}, ça fait combien ?", ["{a} + {b} = {r}.", "Ça fait {r}."]),
        new(Operation.Addition, "Et avec {b} de plus ?", ["{a} + {b} = {r}.", "On arrive à {r}."]),
        new(Operation.Subtraction, "Et si on enlève {b} ?", ["{a} - {b} = {r}.", "Il reste {r}."]),
        new(Operation.Subtraction, "Retire {b} à ce résultat.", ["{a} - {b} = {r}.", "Ça fait {r}."]),
        new(Operation.Subtraction, "Moins {b}, ça fait combien ?", ["{a} - {b} = {r}.", "Ça fait {r}."]),
        new(Operation.Subtraction, "Et si on en retire {b} ?", ["{a} - {b} = {r}.", "Il en reste {r}."]),
        new(Operation.Multiplication, "Multiplie ça par {b}.", ["{a} x {b} = {r}.", "Ça fait {r}."]),
        new(Operation.Multiplication, "Et si on multiplie par {b} ?", ["{a} x {b} = {r}.", "Alors on obtient {r}."]),
        new(Operation.Multiplication, "Et fois {b} ?", ["{a} x {b} = {r}.", "Ça fait {r}."]),
        new(Operation.Multiplication, "Double ce résultat.", ["Le double de {a} est {r}.", "{a} x 2 = {r}."], Fixed: 2),
        new(Operation.Multiplication, "Triple ce résultat.", ["Le triple de {a} est {r}.", "{a} x 3 = {r}."], Fixed: 3),
    ];

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

    /// <summary>Dialogues fabriqués par opération (celle du premier tour).</summary>
    public Dictionary<Operation, int> Counts { get; } = new();

    /// <summary>Dialogues fabriqués par nombre de tours.</summary>
    public SortedDictionary<int, int> TurnCounts { get; } = new();

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

    /// <summary>
    /// count dialogues, dans l'ordre du tirage : 3 sur 10 (MultiTurnShare) ont 2 ou 3 tours, les autres un seul.
    /// </summary>
    public List<Dialogue> Generate(int count, int seed)
    {
        var random = new Random(seed);
        var questions = new HashSet<string>();
        var dialogues = new List<Dialogue>(count);
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

            var turns = new List<Turn> { new(question, Fill(reply, a, b, problem.Result), problem) };
            // 3 conversations sur 10 (les 3 premières de chaque dizaine) : 1 ou 2 questions de plus sur le résultat.
            if (dialogues.Count % 10 < MultiTurnShare * 10)
            {
                int more = random.Next(1, 3);
                while (turns.Count <= more && FollowUpOf(random, turns[^1].Problem.Result) is { } next)
                    turns.Add(next);
            }

            dialogues.Add(new Dialogue(turns));
            Counts[op] = Counts.GetValueOrDefault(op) + 1;
            TurnCounts[turns.Count] = TurnCounts.GetValueOrDefault(turns.Count) + 1;
        }
        return dialogues;
    }

    /// <summary>
    /// Une question sur le résultat précédent : ajouter ou enlever un nombre de 1 à 100 (jamais en dessous de 0),
    /// ou multiplier par 2 à 10 un résultat qui ne dépasse pas 100. null si aucun essai ne convient (calcul de eval.json).
    /// </summary>
    private Turn? FollowUpOf(Random random, int previous)
    {
        for (int attempt = 0; attempt < 20; attempt++)
        {
            var followUp = FollowUps[random.Next(FollowUps.Length)];
            if (followUp.Op == Operation.Multiplication && previous > MaxNumber || followUp.Op == Operation.Subtraction && previous < 1)
                continue;

            int b = followUp.Fixed ?? followUp.Op switch
            {
                Operation.Addition => random.Next(1, MaxNumber + 1),
                Operation.Subtraction => random.Next(1, Math.Min(previous, MaxNumber) + 1),
                _ => random.Next(2, 11),
            };
            var problem = Problem.Of(followUp.Op, previous, b);
            string question = Fill(followUp.Question, previous, b, problem.Result);
            if (_excludedProblems.Contains(problem) || _excludedQuestions.Contains(Simplify(question)))
                continue;

            string reply = followUp.Replies[random.Next(followUp.Replies.Length)];
            return new Turn(question, Fill(reply, previous, b, problem.Result), problem);
        }
        return null;
    }

    /// <summary>Les dialogues au format de dialogues.txt.</summary>
    public static string Format(IEnumerable<Dialogue> dialogues)
    {
        var output = new StringBuilder();
        foreach (var dialogue in dialogues)
        {
            if (output.Length > 0) output.Append('\n');
            foreach (var turn in dialogue.Turns)
                output.Append("Toi: ").Append(turn.Question).Append('\n').Append("IA: ").Append(turn.Reply).Append('\n');
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
