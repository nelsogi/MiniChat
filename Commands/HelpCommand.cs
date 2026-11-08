using MiniChat.UI;
using Spectre.Console;

namespace MiniChat.Commands;

public sealed class HelpCommand : ICommand
{
    public string Name => "help";
    public string Description => "Affiche cette liste de commandes";
    public string Usage => "/help";

    public void Execute(CommandContext context, string[] args)
    {
        var table = new Table()
            .Border(TableBorder.Simple)
            .BorderStyle(Theme.Style(Theme.Border))
            .AddColumn(new TableColumn($"[{Theme.Dim}]Commande[/]").NoWrap())
            .AddColumn(new TableColumn($"[{Theme.Dim}]Ce qu'elle fait[/]"));

        foreach (var command in context.Registry.Commands)
            table.AddRow(new Text(command.Usage, Theme.Style(Theme.Command)), new Text(command.Description));

        AnsiConsole.Write(table);
        Messages.Dim("Écris un message sans « / » pour discuter avec le modèle.");
    }
}
