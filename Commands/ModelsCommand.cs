using System.Globalization;
using MiniChat.UI;
using Spectre.Console;

namespace MiniChat.Commands;

public sealed class ModelsCommand : ICommand
{
    public string Name => "models";
    public string Description => "Liste les modèles sauvegardés : date, taille, loss de vérification";
    public string Usage => "/models";

    public void Execute(CommandContext context, string[] args)
    {
        var models = context.Workbench.ListModels();
        if (models.Count == 0)
        {
            Messages.Warning("Aucun modèle sauvegardé : lance /train pour en entraîner un.");
            return;
        }

        // Tient dans 80 colonnes : la forme du réseau (couches, dimension) est dans /info pour le modèle chargé.
        var table = new Table()
            .Border(TableBorder.Simple)
            .BorderStyle(Theme.Style(Theme.Border))
            .AddColumn(new TableColumn($"[{Theme.Dim}]Nom[/]").NoWrap())
            .AddColumn(new TableColumn($"[{Theme.Dim}]Taille[/]").RightAligned().NoWrap())
            .AddColumn(new TableColumn($"[{Theme.Dim}]Loss[/]").RightAligned().NoWrap())
            .AddColumn(new TableColumn($"[{Theme.Dim}]Pas[/]").RightAligned().NoWrap())
            .AddColumn(new TableColumn($"[{Theme.Dim}]Entraîné[/]").NoWrap())
            .AddColumn(new TableColumn($"[{Theme.Dim}]Sauvegardé[/]").NoWrap());

        foreach (var model in models)
        {
            bool loaded = model.Name == context.Workbench.ModelSource;
            table.AddRow(
                new Markup(loaded
                    ? $"[{Theme.Accent}]{Theme.ReplyBullet} {Markup.Escape(model.Name)}[/]"
                    : $"  [{Theme.Command}]{Markup.Escape(model.Name)}[/]"),
                new Text($"{model.ParameterCount / 1e6:0.0} M"),
                new Text(model.Info is { } info ? info.ValLoss.ToString("F3") : "?"),
                new Text(model.Info is { } steps ? steps.Step.ToString("N0") : "?"),
                new Text(model.Info is { } trained ? RunDate(trained.RunId) : "?"),
                new Text(model.SavedAt.ToString(DateFormat)));
        }

        AnsiConsole.Write(table);
        Messages.Dim($"{Theme.ReplyBullet} = modèle chargé. Loss = loss de vérification notée à la sauvegarde, sur les données " +
                      "de l'époque : pour comparer deux modèles sur les mêmes questions, /compare.");
    }

    private const string DateFormat = "dd'/'MM'/'yy HH:mm";   // '/' : sinon remplacé par le séparateur de la langue

    // RunId = date du début de l'entraînement, "2026-11-05_18-13-15" -> "05/11/26 18:13".
    private static string RunDate(string runId) =>
        DateTime.TryParseExact(runId, "yyyy-MM-dd_HH-mm-ss", CultureInfo.InvariantCulture, DateTimeStyles.None, out var date)
            ? date.ToString(DateFormat)
            : runId;
}
