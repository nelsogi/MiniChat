using MiniChat.App;
using Spectre.Console;

namespace MiniChat.UI;

/// <summary>
/// Le cadre d'accueil de Nel Code : le nom, le modèle chargé (ou comment en avoir un),
/// GPU ou CPU, les dossiers, et l'astuce /help.
/// </summary>
public static class WelcomeBanner
{
    public static void Show(Workbench workbench)
    {
        var grid = new Grid().AddColumn(new GridColumn().PadRight(2)).AddColumn();
        grid.AddRow(Label("Modèle"), new Markup(ModelText(workbench)));
        grid.AddRow(Label("Calcul"), new Markup(workbench.OnGpu
            ? $"[{Theme.Success}]GPU NVIDIA (CUDA)[/]"
            : $"CPU [{Theme.Dim}](pas de GPU NVIDIA utilisable ; si tu en as un, mets à jour son pilote)[/]"));
        grid.AddRow(Label("Données"), new Text(workbench.DataFolder, Theme.Style(Theme.Dim)));
        grid.AddRow(Label("Modèles"), new Text(workbench.ModelsFolder, Theme.Style(Theme.Dim)));
        if (workbench.HasInterruptedTraining)
            grid.AddRow(Label("Reprise"), new Markup(
                $"[{Theme.Warning}]un entraînement interrompu attend :[/] [{Theme.Command}]/resume[/]"));

        var content = new Rows(
            new Markup($"[{Theme.Accent}]{Theme.Logo}[/] Bienvenue dans [bold {Theme.Accent}]Nel Code[/] !"),
            new Text(""),
            grid,
            new Text(""),
            new Markup($"[{Theme.Dim}]Astuce :[/] [{Theme.Command}]/help[/] [{Theme.Dim}]pour les commandes, " +
                       "ou écris directement pour discuter.[/]"));

        AnsiConsole.Write(new Panel(new Padder(content, new Padding(1, 0)))
            .RoundedBorder()
            .BorderStyle(Theme.Style(Theme.Accent)));
        AnsiConsole.WriteLine();
    }

    private static string ModelText(Workbench workbench)
    {
        if (workbench.Model is not { } model)
            return $"[{Theme.Warning}]aucun modèle entraîné[/] [{Theme.Dim}]— tape[/] [{Theme.Command}]/train[/] " +
                   $"[{Theme.Dim}]pour en créer un[/]";

        var config = model.Config;
        string size = $"[bold]{model.ParameterCount / 1e6:0.0} M[/] paramètres [{Theme.Dim}]· {config.Layers} couches · " +
                      $"{config.Heads} têtes · contexte {config.BlockSize} tokens[/]";
        return workbench.ModelSource == Workbench.PretrainedModel
            ? $"{size}\n[{Theme.Warning}]pré-entraîné seulement[/] [{Theme.Dim}]— [/][{Theme.Command}]/finetune[/] " +
              $"[{Theme.Dim}]pour qu'il sache discuter[/]"
            : size;
    }

    private static Markup Label(string text) => new($"[{Theme.Dim}]{text}[/]");
}
