namespace MiniChat.Commands;

public sealed class TrainCommand : ICommand
{
    public string Name => "train";
    public string Description => "Entraîne un nouveau modèle : pré-entraînement sur texte.txt + fine-tuning sur les dialogues";
    public string Usage => "/train";

    public void Execute(CommandContext context, string[] args) => context.Workbench.Train(context.Cancellation);
}
