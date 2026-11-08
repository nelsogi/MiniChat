using MiniChat.App;
using MiniChat.Chat;
using MiniChat.Evaluation;
using MiniChat.UI;
using Spectre.Console;

namespace MiniChat.Commands;

/// <summary>
/// Les questions de test (questions-test.txt) posées à deux modèles, réponses côte à côte.
/// Mêmes conditions pour les deux (température et graine fixes, comme /test) : les différences viennent des modèles.
/// </summary>
public sealed class CompareCommand : ICommand
{
    public string Name => "compare";
    public string Description => "Fait répondre deux modèles aux questions de test, côte à côte";
    public string Usage => "/compare <nom1> <nom2>";

    public void Execute(CommandContext context, string[] args)
    {
        if (args.Length != 2)
            throw new UserException($"Quels modèles ? Utilisation : {Usage}, par exemple : /compare model avant-continue " +
                                    "(/models pour la liste)");

        var (first, second) = (args[0], args[1]);
        var answers = context.Workbench.Compare(first, second, context.Cancellation);

        var models = context.Workbench.ListModels();
        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderStyle(Theme.Style(Theme.Border))
            .Expand()
            .ShowRowSeparators()
            .AddColumn(new TableColumn($"[{Theme.Dim}]Question[/]"))
            .AddColumn(new TableColumn(Header(first, models)))
            .AddColumn(new TableColumn(Header(second, models)));

        for (int i = 0; i < answers.Count; i++)
        {
            var (question, a, b) = answers[i];
            table.AddRow(
                new Text($"{i + 1}. {question}", Theme.Style(Theme.User)),
                new Text(a.Length > 0 ? a : "(rien)"),
                new Text(b.Length > 0 ? b : "(rien)"));
        }

        AnsiConsole.Write(table);
        Messages.Dim($"Réglages des deux côtés : température {QuestionTest.Temperature}, top-k {ChatSession.DefaultTopK}, " +
                     $"top-p {ChatSession.DefaultTopP}, graine {QuestionTest.Seed}.");
        if (context.Cancellation.IsCancellationRequested)
            Messages.Warning($"Arrêté avant la fin (Échap) : {answers.Count} question(s) posée(s).");
    }

    // "model-pretrain · 3,123" : le nom, et la loss de vérification notée à sa sauvegarde.
    private static string Header(string name, IReadOnlyList<SavedModel> models)
    {
        var model = models.FirstOrDefault(m => m.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
        string loss = model?.Info is { } info ? $" [{Theme.Dim}]· loss {info.ValLoss:F3}[/]" : "";
        return $"[{Theme.Accent}]{Markup.Escape(model?.Name ?? name)}[/]{loss}";
    }
}
