using MiniChat.Config;
using MiniChat.Model;
using MiniChat.Storage;
using MiniChat.Tokenization;
using MiniChat.Training;
using MiniChat.UI;
using static TorchSharp.torch;

namespace MiniChat.App;

/// <summary>
/// Phase 1 : apprend la langue au modèle sur texte.txt.
/// continuation non null (--continue) : store n'est remplacé que si la loss de vérification a baissé.
/// </summary>
public sealed class PretrainPhase
{
    private readonly DatasetFactory _data;
    private readonly TrainingPhase _phase;

    public PretrainPhase(DatasetFactory data, TrainingPhase phase)
    {
        _data = data;
        _phase = phase;
    }

    public void Run(MiniGpt model, BpeTokenizer tokenizer, ModelStore store, TrainingConfig config,
        TrainingRun run, CheckpointState? resume, ContinueState? continuation = null, TrainingInfo? previous = null)
    {
        // À la reprise, graine décalée du pas : on ne rejoue pas les mêmes extraits qu'au début.
        // Une continuation part aussi de son premier pas (sinon elle retirerait les extraits de --train).
        int seed = config.Seed + (resume?.Step ?? config.FirstStep);
        manual_seed(seed);

        // using : les tokens sont sur le GPU (~0,9 Go), libérés à la fin de la phase.
        using var dataset = _data.GeneralText(tokenizer, seed);
        Messages.Info($"texte.txt : {dataset.Characters:N0} caractères -> {dataset.TrainLength + dataset.ValidationLength:N0} tokens.");
        config = TrainingPhase.Planned(config, dataset, model, resume);
        var result = _phase.Run("Phase 1 - pré-entraînement", "pretrain", dataset, model, tokenizer, config, run,
            resume, optimizerPath: continuation is null ? null : store.OptimizerPath);

        TrainingPhase.SaveResult(store, ModelCatalog.PretrainedModel, model, tokenizer, result, run.Checkpoint.RunId, continuation, previous);
    }
}
