using MiniChat.UI;
using Spectre.Console;

namespace MiniChat.Commands;

public sealed class ClearCommand : ICommand
{
    public string Name => "clear";
    public string Description => "Efface l'écran et repart d'une conversation vide";
    public string Usage => "/clear";

    public void Execute(CommandContext context, string[] args)
    {
        context.Chat?.Reset();
        AnsiConsole.Clear();
        WelcomeBanner.Show(context.Workbench);
    }
}
