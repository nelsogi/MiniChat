namespace MiniChat.Commands;

public sealed class QuitCommand : ICommand
{
    public string Name => "quit";
    public string Description => "Quitte Nel Code";
    public string Usage => "/quit";

    public void Execute(CommandContext context, string[] args) => context.QuitRequested = true;
}
