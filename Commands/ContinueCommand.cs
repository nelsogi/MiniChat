using MiniChat.App;

namespace MiniChat.Commands;

/// <summary>
/// Continue un modèle déjà entraîné, avec les données actuelles de data/ :
///   /continue 2000              2000 pas de pré-entraînement en plus (comme --continue 2000)
///   /continue 200 finetune      200 pas de fine-tuning en plus (comme --continue-finetune 200)
///   /continue 2000 --lr 2e-4    vitesse de départ choisie (comme --lr)
/// </summary>
public sealed class ContinueCommand : ICommand
{
    public string Name => "continue";
    public string Description => "Entraîne plus longtemps un modèle existant (pré-entraînement, ou fine-tuning avec « finetune »)";
    public string Usage => "/continue <étapes> [finetune] [--lr 3e-4]";

    public void Execute(CommandContext context, string[] args)
    {
        if (args.Length == 0)
            throw new UserException($"Combien d'étapes ? Utilisation : {Usage}");

        double steps = CommandContext.ParseNumber(args[0], Usage);
        if (steps <= 0 || steps != Math.Floor(steps) || steps > int.MaxValue / 2)
            throw new UserException("Il faut un nombre entier d'étapes, par exemple : /continue 2000");

        string phase = "pretrain";
        double? learningRate = null;
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "finetune":
                    phase = "finetune";
                    break;
                case "pretrain":
                    phase = "pretrain";
                    break;
                case "--lr" when i + 1 < args.Length:
                    learningRate = CommandContext.ParseNumber(args[++i], Usage);
                    if (learningRate <= 0)
                        throw new UserException("--lr attend un nombre positif, par exemple : --lr 3e-4");
                    break;
                default:
                    throw new UserException($"« {args[i]} » : option inconnue. Utilisation : {Usage}");
            }
        }

        context.Workbench.Continue(phase, (int)steps, learningRate, context.Cancellation);
    }
}
