using MiniChat.Config;
using MiniChat.Data;
using MiniChat.Evaluation;
using MiniChat.Generation;
using MiniChat.Model;
using MiniChat.Storage;
using MiniChat.Tokenization;
using MiniChat.Training;
using MiniChat.UI;
using static TorchSharp.torch;

namespace MiniChat.App;

/// <summary>
/// Un entraînement en cours : où il écrit (CSV des loss, checkpoint, réponses aux questions de test)
/// et l'arrêt demandé (Échap) : la phase en cours s'arrête proprement, la suivante n'est pas lancée.
/// </summary>
public sealed record TrainingRun(TrainingLog Log, Checkpoint Checkpoint, string AnswersPath, CancellationToken Cancellation);

/// <summary>Où un entraînement écrit son suivi : le CSV des loss, et les questions pour les exemples.</summary>
public sealed record TrainingLog(string LossCsvPath, string QuestionsPath);

/// <summary>
/// Ce que les deux phases font pareil : faire tourner le Trainer avec sa barre de progression,
/// fixer le dernier pas, et sauvegarder le résultat.
/// </summary>
public sealed class TrainingPhase
{
    private readonly Device _device;

    public TrainingPhase(Device device) => _device = device;

    public TrainingResult Run(string name, string phase, TextDataset dataset, MiniGpt model, BpeTokenizer tokenizer,
        TrainingConfig config, TrainingRun run, CheckpointState? resume,
        string? optimizerPath = null, TextDataset? generalText = null)
    {
        // Lu dès le début : si le fichier de questions manque, on le sait tout de suite, pas après des heures.
        string sampleQuestion = QuestionTest.ReadQuestions(run.Log.QuestionsPath)[0];
        var generator = new TextGenerator(model, tokenizer, _device);

        Messages.Title(name);
        Messages.Info($"{dataset.TrainLength + dataset.ValidationLength:N0} tokens, {config.Steps - config.FirstStep} étapes.");
        return TrainingProgressView.Run(config.FirstStep, config.Steps, progress =>
            new Trainer(model, dataset, config, _device, run.Checkpoint, progress, generalText, phase, run.Log.LossCsvPath,
                showSample: step => progress.Log(QuestionTest.Sample(generator, sampleQuestion, step)))
                .Train(resume, optimizerPath, run.Cancellation));
    }

    // Le dernier pas de la phase : celui de l'entraînement repris (même fin prévue, donc même cosinus), sinon
    // calculé depuis Epochs et la taille des données d'entraînement. Une continuation garde le sien (Epochs null).
    public static TrainingConfig Planned(TrainingConfig config, TextDataset dataset, MiniGpt model, CheckpointState? resume) =>
        resume?.Steps is { } steps ? config with { Steps = steps } : config.ForData(dataset.TrainLength, model.Config.BlockSize);

    // Sauvegarde le modèle avec l'état d'AdamW et l'avancement (pas atteint, loss, date du CSV).
    // Une continuation ne remplace l'ancien modèle que si sa loss de vérification est meilleure. Les deux loss sont
    // mesurées sur les données actuelles (l'ancien modèle au premier pas) : celle notée à l'ancienne sauvegarde
    // a pu l'être sur d'autres données, elle n'est affichée que pour info.
    public static void SaveResult(ModelStore store, string name, MiniGpt model, BpeTokenizer tokenizer, TrainingResult result,
        string runId, ContinueState? continuation, TrainingInfo? previous)
    {
        if (continuation is not null)
        {
            string saved = previous is null ? "" : $", {previous.ValLoss:F3} à sa sauvegarde";
            Messages.Info($"Loss de vérification : ancienne {result.StartValLoss:F3} (sur les données actuelles{saved}), " +
                          $"nouvelle {result.BestValLoss:F3} (pas {result.BestStep}).");
            if (!result.Improved)
            {
                Messages.Warning($"Pas meilleure : '{name}' n'est pas remplacé, l'ancien modèle est gardé.");
                return;
            }
        }

        // ponytail: l'état d'AdamW est celui du dernier pas, les poids ceux du meilleur pas ;
        // assez proches pour repartir, pas la peine de sauvegarder l'optimiseur à chaque meilleur pas.
        store.Save(model, tokenizer, result.OptimizerStatePath, new TrainingInfo(result.LastStep, result.BestValLoss, runId));
        Messages.Success(continuation is null
            ? $"Modèle sauvegardé dans '{name}'."
            : $"Meilleure : modèle remplacé dans '{name}'.");
    }
}
