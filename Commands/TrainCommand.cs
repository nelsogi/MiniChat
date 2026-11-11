using MiniChat.App;
using MiniChat.Config;
using MiniChat.UI;

namespace MiniChat.Commands;

/// <summary>
/// Nouveau modèle : pré-entraînement puis fine-tuning.
///   /train                 modèle small, 1,1 passage sur texte.txt (le nombre de pas est calculé depuis sa taille)
///   /train --size medium   un plus gros modèle (tailles : ModelConfig.Sizes)
///   /train --epochs 0,5    un demi-passage (plus court, pour essayer)
/// Montre d'abord la taille du modèle et la durée estimée, et demande confirmation.
/// </summary>
public sealed class TrainCommand : ICommand
{
    public string Name => "train";
    public string Description => "Entraîne un nouveau modèle : pré-entraînement sur texte.txt + fine-tuning sur les dialogues";
    public string Usage => $"/train [--size {string.Join("|", ModelConfig.Sizes.Keys)}] [--epochs 1,1]";

    public void Execute(CommandContext context, string[] args)
    {
        var (size, epochs) = ParseOptions(args);
        var plan = context.Workbench.Plan(size, epochs);
        var model = plan.Config;
        var pretrain = plan.Pretraining;

        Messages.Info($"Modèle {plan.Size} : {model.EmbedDim} dimensions, {model.Layers} couches, {model.Heads} têtes, " +
                      $"contexte de {model.BlockSize} tokens, {model.VocabSize:N0} tokens de vocabulaire, {model.Architecture}.");
        Messages.Info($"  {plan.Parameters:N0} paramètres.");
        Messages.Info($"  Pré-entraînement : ~{plan.PretrainSteps:N0} étapes de {pretrain.BatchSize} x {pretrain.EffectiveBatchSize / pretrain.BatchSize} " +
                      $"extraits ({(long)pretrain.EffectiveBatchSize * model.BlockSize:N0} tokens), vitesse max {pretrain.LearningRate:0.0e0}, " +
                      $"warmup {pretrain.WarmupSteps} pas : ~{Messages.Duration(plan.PretrainDuration)}.");
        Messages.Info($"  Fine-tuning : ~{plan.FineTuneSteps:N0} étapes : ~{Messages.Duration(plan.FineTuneDuration)}.");
        Messages.Info($"  Total estimé : ~{Messages.Duration(plan.PretrainDuration + plan.FineTuneDuration)}, " +
                      "plus quelques minutes pour le tokenizer et le découpage des fichiers en tokens.");
        Messages.Dim("  (estimation d'après la taille des fichiers et la vitesse mesurée sur une RTX 3060 Ti ; interruptible avec Échap, /resume)");

        if (!context.Confirm("Lancer l'entraînement ?"))
        {
            Messages.Dim("Entraînement annulé.");
            return;
        }
        context.Workbench.Train(size, epochs, context.Cancellation);
    }

    // "--size medium" et "--epochs 1,5", dans n'importe quel ordre, chacun facultatif.
    private (string Size, double? Epochs) ParseOptions(string[] args)
    {
        string size = ModelConfig.DefaultSize;
        double? epochs = null;
        if (args.Length % 2 != 0)
            throw new UserException($"Option sans valeur. Utilisation : {Usage}");

        for (int i = 0; i < args.Length; i += 2)
        {
            if (args[i].Equals("--size", StringComparison.OrdinalIgnoreCase))
                size = args[i + 1].ToLowerInvariant();
            else
                epochs = CommandContext.ParseEpochs(args[i..(i + 2)], Usage);
        }
        return (size, epochs);
    }
}
