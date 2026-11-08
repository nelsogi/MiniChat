namespace MiniChat.Commands;

public sealed class FinetuneCommand : ICommand
{
    public string Name => "finetune";
    public string Description => "Refait seulement le fine-tuning, depuis le modèle pré-entraîné (model-pretrain)";
    public string Usage => "/finetune";

    public void Execute(CommandContext context, string[] args) => context.Workbench.FineTuneOnly();
}
