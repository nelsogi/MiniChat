using MiniChat.UI;

namespace MiniChat.Commands;

public sealed class ResetCommand : ICommand
{
    public string Name => "reset";
    public string Description => "Oublie la conversation (le modèle, lui, ne change pas)";
    public string Usage => "/reset";

    public void Execute(CommandContext context, string[] args)
    {
        context.RequireChat().Reset();
        Messages.Dim("(conversation oubliée)");
    }
}
