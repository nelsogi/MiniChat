namespace MiniChat.Commands;

public sealed class ResumeCommand : ICommand
{
    public string Name => "resume";
    public string Description => "Reprend un entraînement interrompu (coupure, plantage, Ctrl+C) là où il s'est arrêté";
    public string Usage => "/resume";

    public void Execute(CommandContext context, string[] args) => context.Workbench.Resume(context.Cancellation);
}
