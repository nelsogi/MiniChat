using MiniChat.Config;
using MiniChat.Model;
using MiniChat.Storage;
using MiniChat.Tokenization;
using MiniChat.Training;
using MiniChat.UI;

namespace MiniChat.App;

/// <summary>
/// --continue / --continue-finetune : steps pas de plus sur un modèle déjà entraîné, avec les données actuelles
/// de data/ et l'optimiseur sauvegardé avec lui. Les pas continuent la numérotation (et le CSV) d'avant.
/// </summary>
public sealed class ContinuePhase
{
    private readonly PretrainPhase _pretrain;
    private readonly FinetunePhase _finetune;

    public ContinuePhase(PretrainPhase pretrain, FinetunePhase finetune)
    {
        _pretrain = pretrain;
        _finetune = finetune;
    }

    public void Run(ContinueState continuation, MiniGpt model, BpeTokenizer tokenizer, ModelStore store,
        TrainingInfo? previous, TrainingRun run, CheckpointState? resume)
    {
        var config = BaseConfig(continuation.Phase, model.Config).Continuation(continuation.FirstStep, continuation.Steps, continuation.LearningRate);
        Messages.Info($"Continuation : pas {config.FirstStep} -> {config.Steps}, warmup {config.WarmupSteps} pas, " +
                      $"vitesse {config.LearningRate:0.0e0} -> {config.MinLearningRate:0.0e0}.");

        if (continuation.Phase == "pretrain")
            _pretrain.Run(model, tokenizer, store, config, run, resume, continuation, previous);
        else
            _finetune.Run(model, tokenizer, store, config, run, resume, continuation, previous);
    }

    /// <summary>Les réglages de la phase continuée, avant Continuation.</summary>
    public static TrainingConfig BaseConfig(string phase, ModelConfig model) =>
        phase == "pretrain" ? TrainingConfig.PretrainingFor(model) : TrainingConfig.FineTuningFor(model);
}
