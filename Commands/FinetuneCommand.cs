namespace MiniChat.Commands;

/// <summary>
/// Refait le fine-tuning depuis model-pretrain :
///   /finetune                2,5 passages sur les dialogues (le nombre de pas est calculé depuis leur taille)
///   /finetune --epochs 3     3 passages
/// </summary>
public sealed class FinetuneCommand : ICommand
{
    public string Name => "finetune";
    public string Description => "Refait seulement le fine-tuning, depuis le modèle pré-entraîné (model-pretrain)";
    public string Usage => "/finetune [--epochs 2,5]";

    public void Execute(CommandContext context, string[] args) =>
        context.Workbench.FineTuneOnly(CommandContext.ParseEpochs(args, Usage), context.Cancellation);
}
