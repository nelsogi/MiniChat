using MiniChat.Config;
using MiniChat.Model;
using MiniChat.Storage;
using MiniChat.Tokenization;
using MiniChat.Training;
using static TorchSharp.torch;

namespace MiniChat.App;

/// <summary>
/// Phase 2 : spécialise le modèle (déjà pré-entraîné) sur les dialogues.
/// Le modèle n'est noté que sur les réponses "IA:", et 20 % de chaque batch vient de texte.txt
/// pour qu'il n'oublie pas le français général.
/// </summary>
public sealed class FinetunePhase
{
    private readonly DatasetFactory _data;
    private readonly TrainingPhase _phase;

    public FinetunePhase(DatasetFactory data, TrainingPhase phase)
    {
        _data = data;
        _phase = phase;
    }

    public void Run(MiniGpt model, BpeTokenizer tokenizer, ModelStore store, TrainingConfig config,
        TrainingRun run, CheckpointState? resume, ContinueState? continuation = null, TrainingInfo? previous = null)
    {
        int seed = config.Seed + (resume?.Step ?? config.FirstStep);
        manual_seed(seed);

        using var dialogues = _data.Dialogues(tokenizer, seed);
        // Autre graine : les extraits de texte général ne suivent pas le même tirage que les dialogues.
        using var generalText = _data.GeneralText(tokenizer, seed + 1);
        config = TrainingPhase.Planned(config, dialogues, model, resume);

        var result = _phase.Run($"Phase 2 - fine-tuning (loss sur les réponses IA, {config.GeneralTextRatio:P0} de texte général)",
            "finetune", dialogues, model, tokenizer, config, run, resume,
            optimizerPath: continuation is null ? null : store.OptimizerPath, generalText);

        TrainingPhase.SaveResult(store, ModelCatalog.ChatModel, model, tokenizer, result, run.Checkpoint.RunId, continuation, previous);
    }
}
