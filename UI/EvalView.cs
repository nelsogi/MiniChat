using MiniChat.App;
using MiniChat.Evaluation;
using Spectre.Console;

namespace MiniChat.UI;

/// <summary>Les scores de /eval (avec l'évaluation d'avant) et de /compare (deux modèles côte à côte), par catégorie.</summary>
public static class EvalView
{
    /// <summary>Le score de chaque catégorie et le total, avec ceux de l'évaluation d'avant du même modèle.</summary>
    public static void Show(EvalRun run)
    {
        var previous = run.Previous;
        var table = Table($"[{Theme.Dim}]Catégorie[/]", $"[{Theme.Accent}]{Markup.Escape(run.Model)}[/]");
        if (previous is not null)
            table.AddColumn(new TableColumn($"[{Theme.Dim}]Avant ({previous.Date:dd'/'MM HH:mm})[/]").RightAligned());

        foreach (var score in run.Result.Scores)
        {
            var scores = new List<string> { Score(score.Correct, score.Total) };
            if (previous is not null)
                scores.Add(previous.Scores.FirstOrDefault(s => s.Category == score.Category) is { } before
                    ? Score(before.Correct, before.Total)
                    : "");
            AddRow(table, score.Category, [.. scores]);
        }

        var total = new List<string> { Score(run.Result.Correct, run.Result.Total) };
        if (previous is not null)
            total.Add(Score(previous.Correct, previous.Total));
        AddRow(table, "total", [.. total], bold: true);

        AnsiConsole.Write(table);
        if (run.History.Count >= 2)
            Messages.Dim($"Historique de '{run.Model}' : " + string.Join(" → ", run.History.TakeLast(HistoryShown - 1)
                .Select(r => $"{r.Correct} ({r.Date:dd'/'MM})").Append($"{run.Result.Correct} (maintenant)")));
        Footer(run.Result);
        Messages.Dim($"Détail (chaque réponse) : {run.ReportPath}");
    }

    // Les dernières évaluations montrées sous le tableau de /eval (tout l'historique : results/eval-<nom>.jsonl).
    private const int HistoryShown = 8;

    /// <summary>Les scores de deux modèles côte à côte (/compare).</summary>
    public static void ShowComparison(EvalRun first, EvalRun second)
    {
        Messages.Title("Évaluation notée (data/eval.json)");
        var table = Table($"[{Theme.Dim}]Catégorie[/]", $"[{Theme.Accent}]{Markup.Escape(first.Model)}[/]",
            $"[{Theme.Accent}]{Markup.Escape(second.Model)}[/]");
        var secondScores = second.Result.Scores.ToDictionary(s => s.Category);
        foreach (var score in first.Result.Scores)
            AddRow(table, score.Category, [Score(score.Correct, score.Total),
                secondScores.TryGetValue(score.Category, out var other) ? Score(other.Correct, other.Total) : ""]);
        AddRow(table, "total", [Score(first.Result.Correct, first.Result.Total), Score(second.Result.Correct, second.Result.Total)],
            bold: true);

        AnsiConsole.Write(table);
        Footer(first.Result);
        Messages.Dim($"Détail : {first.ReportPath}");
        Messages.Dim($"         {second.ReportPath}");
    }

    private static Table Table(params string[] headers)
    {
        var table = new Table().Border(TableBorder.Simple).BorderStyle(Theme.Style(Theme.Border));
        table.AddColumn(new TableColumn(headers[0]).NoWrap());
        foreach (string header in headers.Skip(1))
            table.AddColumn(new TableColumn(header).RightAligned().NoWrap());
        return table;
    }

    private static void AddRow(Table table, string category, string[] scores, bool bold = false)
    {
        string style = bold ? "bold" : Theme.Text;
        table.AddRow([new Markup($"[{style}]{Markup.Escape(category)}[/]"), .. scores.Select(s => new Markup($"[{style}]{Markup.Escape(s)}[/]"))]);
    }

    private static void Footer(EvalResult result)
    {
        Messages.Dim($"Réponses gloutonnes (toujours le token le plus probable), {EvalTest.MaxReplyTokens} tokens au plus ; " +
                     "juste si la réponse contient une réponse acceptée (sans majuscules ni accents, en mots entiers).");
        if (!result.Complete)
            Messages.Warning($"Arrêtée avant la fin (Échap) : {result.Total} question(s) sur {result.QuestionCount}, " +
                             "pas gardée dans l'historique.");
    }

    // "17 / 20 (85 %)"
    private static string Score(int correct, int total) =>
        total == 0 ? "-" : $"{correct} / {total} ({100.0 * correct / total:0} %)";
}
