using MiniChat.UI;

namespace MiniChat.Commands;

public sealed class TopKCommand : ICommand
{
    public string Name => "topk";
    public string Description => "Top-k : tirage parmi les K tokens les plus probables (0 = désactivé)";
    public string Usage => "/topk [20]";

    public void Execute(CommandContext context, string[] args)
    {
        var chat = context.RequireChat();
        if (args.Length > 0)
            chat.TopK = Math.Max(0, (int)CommandContext.ParseNumber(args[0], Usage));
        Messages.Dim(chat.TopK == 0 ? "(top-k désactivé)" : $"(top-k = {chat.TopK})");
    }
}
