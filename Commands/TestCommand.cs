namespace MiniChat.Commands;

public sealed class TestCommand : ICommand
{
    public string Name => "test";
    public string Description => "Pose les questions de questions-test.txt au modèle chargé (réponses dans results/)";
    public string Usage => "/test";

    public void Execute(CommandContext context, string[] args) => context.Workbench.RunTest(context.Cancellation);
}
