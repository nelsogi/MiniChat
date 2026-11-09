namespace MiniChat.Commands;

/// <summary>
/// Nouveau modèle : pré-entraînement puis fine-tuning.
///   /train                 1,1 passage sur texte.txt (le nombre de pas est calculé depuis sa taille)
///   /train --epochs 0,5    un demi-passage (plus court, pour essayer)
/// </summary>
public sealed class TrainCommand : ICommand
{
    public string Name => "train";
    public string Description => "Entraîne un nouveau modèle : pré-entraînement sur texte.txt + fine-tuning sur les dialogues";
    public string Usage => "/train [--epochs 1,1]";

    public void Execute(CommandContext context, string[] args) =>
        context.Workbench.Train(CommandContext.ParseEpochs(args, Usage), context.Cancellation);
}
