using System.Diagnostics;
using MiniChat.Config;
using MiniChat.Model;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Training;

/// <summary>
/// Une mesure de SpeedBenchmark : tokens appris par seconde avec ce micro-batch, et la mémoire GPU de toute la carte
/// après les pas mesurés (voir GpuMemory). Problem : pourquoi il n'y a pas de vitesse (mémoire dépassée...).
/// </summary>
public sealed record SpeedMeasure(string Phase, int MicroBatch, double? TokensPerSecond, long? GpuMb, string? Problem = null);

/// <summary>
/// Mesure la vitesse d'entraînement d'une taille de modèle selon la taille du micro-batch (BatchSize), à
/// EffectiveBatchSize égal : mêmes tokens par étape, seul le découpage en petits batchs change. Pas de données :
/// des tokens au hasard (la vitesse ne dépend pas du texte), et pour le fine-tuning des conversations de ~100 tokens
/// mises bout à bout (masque par document, voir DocumentMask). Pas d'écriture sur le disque.
///
/// Chaque mesure tourne dans son propre processus (measure, voir Program : --benchmark-measure) : libtorch garde la
/// mémoire qu'il a prise une fois, et après un micro-batch trop gros la mesure suivante partirait d'une carte pleine.
/// Un micro-batch qui dépasse GpuMemoryLimitMb ne compte pas : le Trainer le diviserait par deux au premier pas.
/// </summary>
public static class SpeedBenchmark
{
    private const int WarmupSteps = 2;      // allocations, choix des noyaux cuDNN : pas mesurés
    private const int MeasuredSteps = 6;
    private const int ConversationTokens = 100;

    /// <param name="measure">Une mesure (phase, micro-batch), faite dans un processus à part.</param>
    public static IReadOnlyList<SpeedMeasure> Run(IReadOnlyList<int> microBatches, Func<string, int, SpeedMeasure> measure,
        Action<string> log)
    {
        var results = new List<SpeedMeasure>();
        foreach (string phase in new[] { "pretrain", "finetune" })
        {
            bool full = false;
            foreach (int microBatch in microBatches)
            {
                var result = full
                    ? new SpeedMeasure(phase, microBatch, null, null, "pas essayé (le précédent manquait déjà de mémoire)")
                    : measure(phase, microBatch);
                log($"{phase}, micro-batch {microBatch} : " + (result.TokensPerSecond is { } speed
                    ? $"{speed:N0} tokens/s, {result.GpuMb:N0} Mo" : result.Problem));
                results.Add(result);
                full = result.Problem is not null;
            }
        }
        return results;
    }

    /// <summary>
    /// Une mesure : phase "pretrain" ou "finetune" avec ses réglages (TrainingConfig.PretrainingFor, FineTuningFor),
    /// en micro-batchs de microBatch extraits. À faire dans un processus neuf (voir Run).
    /// </summary>
    public static SpeedMeasure Measure(string phase, ModelConfig config, int microBatch, Device device)
    {
        bool documents = phase == "finetune";
        var training = (documents ? TrainingConfig.FineTuningFor(config) : TrainingConfig.PretrainingFor(config))
            with { BatchSize = microBatch };
        int accumulation = training.EffectiveBatchSize / training.BatchSize;

        manual_seed(training.Seed);
        using var model = new MiniGpt(config).to(device);
        model.train();
        model.DropoutRate.Value = training.Dropout ?? config.Dropout;
        var optimizer = optim.AdamW(model.parameters(), lr: training.LearningRate, beta1: training.Beta1, beta2: training.Beta2,
            weight_decay: training.WeightDecay);

        try
        {
            var clock = new Stopwatch();
            for (int step = 0; step < WarmupSteps + MeasuredSteps; step++)
            {
                if (step == WarmupSteps)
                {
                    Synchronize(device);
                    clock.Start();
                }

                using var scope = NewDisposeScope();
                optimizer.zero_grad();
                for (int micro = 0; micro < accumulation; micro++)
                {
                    using var microScope = NewDisposeScope();
                    var x = randint(config.VocabSize, [training.BatchSize, config.BlockSize], device: device);
                    var y = randint(config.VocabSize, [training.BatchSize, config.BlockSize], device: device);
                    // Une nouvelle conversation à chaque token avec une chance sur ConversationTokens.
                    var docs = documents
                        ? rand([training.BatchSize, config.BlockSize], device: device).lt(1.0 / ConversationTokens)
                            .to(ScalarType.Int64).cumsum(1)
                        : null;
                    var logits = model.Forward(x, cache: null, docs);
                    var loss = functional.cross_entropy(logits.view(-1, config.VocabSize), y.view(-1)) / accumulation;
                    loss.backward();

                    // Dès le premier micro-batch : au-delà de la mémoire de la carte, Windows ne renvoie pas "out of
                    // memory" mais déborde dans la mémoire du PC, des dizaines de fois plus lente (8 sur le modèle
                    // large : pas fini en 30 min). On s'arrête là, sans attendre les pas mesurés.
                    if (step == 0 && micro == 0 && GpuMemory.Read()?.UsedMb is { } used && used > training.GpuMemoryLimitMb)
                        return new SpeedMeasure(phase, microBatch, null, used,
                            $"{used:N0} Mo dès le premier micro-batch : au-delà de la limite de {training.GpuMemoryLimitMb:N0} Mo " +
                            "(GpuMemoryLimitMb), la carte déborde dans la mémoire du PC");
                }
                nn.utils.clip_grad_norm_(model.parameters(), training.GradClip);
                optimizer.step();
            }
            Synchronize(device);
            clock.Stop();

            long tokens = (long)MeasuredSteps * training.EffectiveBatchSize * config.BlockSize;
            long? memory = GpuMemory.Read()?.UsedMb;
            return memory > training.GpuMemoryLimitMb
                ? new SpeedMeasure(phase, microBatch, null, memory,
                    $"{memory:N0} Mo : au-delà de la limite de {training.GpuMemoryLimitMb:N0} Mo (GpuMemoryLimitMb)")
                : new SpeedMeasure(phase, microBatch, tokens / clock.Elapsed.TotalSeconds, memory);
        }
        catch (Exception e) when (e.Message.Contains("out of memory", StringComparison.OrdinalIgnoreCase))
        {
            return new SpeedMeasure(phase, microBatch, null, GpuMemory.Read()?.UsedMb, "mémoire GPU insuffisante (out of memory)");
        }
    }

    private static void Synchronize(Device device)
    {
        if (device.type == DeviceType.CUDA)
            cuda.synchronize();
    }
}
