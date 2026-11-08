using MiniChat.App;
using MiniChat.UI;

namespace MiniChat.Commands;

public sealed class TempCommand : ICommand
{
    public string Name => "temp";
    public string Description => "Température : plus bas = réponses sages, plus haut = créatives (0.1 à 2), pour cette conversation";
    public string Usage => "/temp [0.8]";

    public void Execute(CommandContext context, string[] args)
    {
        var chat = context.RequireChat();
        if (args.Length > 0)
            chat.Temperature = Settings.ClampTemperature(CommandContext.ParseNumber(args[0], Usage));
        Messages.Dim($"(température = {chat.Temperature})");
    }
}
