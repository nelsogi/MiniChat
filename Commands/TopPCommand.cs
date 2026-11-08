using MiniChat.UI;

namespace MiniChat.Commands;

public sealed class TopPCommand : ICommand
{
    public string Name => "topp";
    public string Description => "Top-p : tirage parmi les tokens qui font cette part des chances (1 = désactivé)";
    public string Usage => "/topp [0.9]";

    public void Execute(CommandContext context, string[] args)
    {
        var chat = context.RequireChat();
        if (args.Length > 0)
            chat.TopP = Math.Clamp(CommandContext.ParseNumber(args[0], Usage), 0.05, 1.0);
        Messages.Dim(chat.TopP >= 1.0 ? "(top-p désactivé)" : $"(top-p = {chat.TopP})");
    }
}
