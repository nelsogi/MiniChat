using System.Globalization;
using System.Text;
using System.Text.Json;
using MiniChat.Chat;
using MiniChat.Generation;

namespace MiniChat.Evaluation;

/// <summary>Une question de data/eval.json : sa catégorie, et les réponses acceptées.</summary>
public sealed record EvalQuestion(string Category, string Question, IReadOnlyList<string> Answers);

/// <summary>La réponse du modèle à une question, et si elle est juste.</summary>
public sealed record EvalAnswer(EvalQuestion Question, string Reply, bool Correct);

/// <summary>Le score d'une catégorie : réponses justes sur questions posées.</summary>
public sealed record EvalScore(string Category, int Correct, int Total);

/// <summary>Les réponses à toutes les questions posées (moins que toutes si l'évaluation a été arrêtée).</summary>
public sealed record EvalResult(IReadOnlyList<EvalAnswer> Answers, int QuestionCount)
{
    public int Correct => Answers.Count(a => a.Correct);
    public int Total => Answers.Count;

    /// <summary>Arrêtée avant la fin (Échap) : le score ne compte que les questions posées.</summary>
    public bool Complete => Answers.Count == QuestionCount;

    /// <summary>Le score de chaque catégorie, dans l'ordre du fichier.</summary>
    public IReadOnlyList<EvalScore> Scores => Answers
        .GroupBy(a => a.Question.Category)
        .Select(g => new EvalScore(g.Key, g.Count(a => a.Correct), g.Count()))
        .ToList();
}

/// <summary>
/// Un examen de culture générale qui se note tout seul : les questions de data/eval.json (capitales, sciences,
/// histoire, vocabulaire, calcul), chacune avec ses réponses acceptées. Le modèle répond en glouton (toujours le
/// token le plus probable : pas de hasard, même modèle = mêmes réponses), et une réponse est juste si elle contient
/// l'une des réponses acceptées, sans tenir compte des majuscules ni des accents (voir IsCorrect).
/// Contrairement à questions-test.txt (à lire soi-même), le score se compare d'un modèle à l'autre (voir EvalHistory).
/// </summary>
public static class EvalTest
{
    /// <summary>Longueur max d'une réponse : la bonne réponse vient au début, la suite ne ferait que tenter sa chance.</summary>
    public const int MaxReplyTokens = 48;

    private sealed record FileQuestion(string Categorie, string Question, string[] Reponses);
    private sealed record EvalFile(FileQuestion[] Questions);

    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    public static IReadOnlyList<EvalQuestion> ReadQuestions(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Fichier d'évaluation introuvable : {path}");

        var file = JsonSerializer.Deserialize<EvalFile>(File.ReadAllText(path), JsonOptions);
        var questions = (file?.Questions ?? [])
            .Where(q => !string.IsNullOrWhiteSpace(q.Question) && q.Reponses is { Length: > 0 })
            .Select(q => new EvalQuestion(q.Categorie ?? "", q.Question.Trim(), q.Reponses))
            .ToList();
        return questions.Count > 0 ? questions : throw new InvalidDataException($"Aucune question dans {path}");
    }

    /// <summary>
    /// La réponse contient-elle l'une des réponses acceptées ? Les deux sont d'abord simplifiées (voir Normalize),
    /// puis comparées en mots entiers : « 4 » n'est pas trouvé dans « 14 », ni « lent » dans « excellent ».
    /// </summary>
    public static bool IsCorrect(string reply, IEnumerable<string> answers)
    {
        string text = $" {Normalize(reply)} ";
        return answers.Select(Normalize).Any(answer => answer.Length > 0 && text.Contains($" {answer} ", StringComparison.Ordinal));
    }

    /// <summary>
    /// Minuscules, sans accents (« Pékin » -> « pekin », « cœur » -> « coeur », « H₂O » -> « h2o »), et tout ce qui
    /// n'est ni lettre ni chiffre devient une espace (« Marie-Antoinette » -> « marie antoinette »).
    /// </summary>
    internal static string Normalize(string text)
    {
        var simple = new StringBuilder(text.Length);
        foreach (char c in text.Replace("œ", "oe").Replace("Œ", "oe").Replace("æ", "ae").Replace("Æ", "ae")
                     .Normalize(NormalizationForm.FormKD))
        {
            if (CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.NonSpacingMark)
                continue;   // l'accent, séparé de sa lettre par FormKD
            simple.Append(char.IsLetterOrDigit(c) ? char.ToLowerInvariant(c) : ' ');
        }
        return string.Join(' ', simple.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries));
    }

    /// <summary>La réponse gloutonne à une question, sans historique : toujours le token le plus probable.</summary>
    public static string Answer(TextGenerator generator, string question)
    {
        // Avec une graine, le tirage (déjà joué d'avance avec top-k 1) ne touche pas au hasard global.
        var chat = new ChatSession(generator) { Temperature = 1, TopK = 1, TopP = 1, MaxReplyTokens = MaxReplyTokens, Seed = 0 };
        return chat.Send(question, _ => { }).Trim();
    }

    /// <summary>
    /// Pose les questions une par une. onQuestion(i) : avant la question n° i (pour l'affichage).
    /// cancellation : s'arrête avant la question suivante, le résultat garde les réponses déjà données.
    /// </summary>
    public static EvalResult Run(TextGenerator generator, IReadOnlyList<EvalQuestion> questions,
        Action<int>? onQuestion = null, CancellationToken cancellation = default)
    {
        var answers = new List<EvalAnswer>();
        for (int i = 0; i < questions.Count && !cancellation.IsCancellationRequested; i++)
        {
            onQuestion?.Invoke(i);
            string reply = Answer(generator, questions[i].Question);
            answers.Add(new EvalAnswer(questions[i], reply, IsCorrect(reply, questions[i].Answers)));
        }
        return new EvalResult(answers, questions.Count);
    }

    /// <summary>Le détail : chaque question, la réponse du modèle, juste ou non, et ce qui était accepté.</summary>
    public static string Report(EvalResult result, string description)
    {
        var report = new StringBuilder();
        report.AppendLine($"MiniChat - évaluation (/eval) - {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        report.AppendLine(description);
        report.AppendLine($"Score : {result.Correct} / {result.Total}" + (result.Complete ? "" : $" (arrêtée : {result.Total} questions sur {result.QuestionCount})"));
        foreach (var score in result.Scores)
            report.AppendLine($"  {score.Category} : {score.Correct} / {score.Total}");
        report.AppendLine();

        for (int i = 0; i < result.Answers.Count; i++)
        {
            var answer = result.Answers[i];
            report.AppendLine($"{i + 1,3}. [{answer.Question.Category}] {answer.Question.Question}");
            report.AppendLine($"     {(answer.Correct ? "juste" : "faux ")} | {ChatSession.BotPrefix}{answer.Reply}");
            if (!answer.Correct)
                report.AppendLine($"             attendu : {string.Join(" / ", answer.Question.Answers)}");
        }
        return report.ToString();
    }
}
