using System.Diagnostics;
using MiniChat.Config;
using MiniChat.Data;
using MiniChat.Model;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Training;

/// <summary>
/// Bilan d'une phase. StartValLoss = loss de vérification du modèle de départ (au premier pas),
/// BestValLoss/BestStep = les poids gardés, LastStep = pas atteint (arrêt anticipé ou fin).
/// OptimizerStatePath = état d'AdamW à la fin, à sauvegarder avec le modèle pour pouvoir continuer (--continue).
/// </summary>
public sealed record TrainingResult(float StartValLoss, float BestValLoss, int BestStep, int LastStep,
    string OptimizerStatePath)
{
    /// <summary>Les poids gardés sont meilleurs que ceux du départ (sinon ce sont les mêmes).</summary>
    public bool Improved => BestValLoss < StartValLoss;
}

/// <summary>
/// La boucle d'apprentissage. À chaque étape :
///   1. le modèle prédit le caractère suivant pour chaque position,
///   2. on mesure son erreur (la "loss", cross-entropy),
///   3. backward() calcule comment chaque poids a contribué à l'erreur,
///   4. l'optimiseur AdamW ajuste un peu chaque poids pour réduire l'erreur.
///
/// En plus :
///   - la vitesse d'apprentissage suit un warmup puis une descente en cosinus,
///   - on garde une copie des poids quand la loss de vérification est la plus basse (sur disque,
///     checkpoint/best.dat), et on la remet dans le modèle à la fin (le modèle sauvegardé = le meilleur vu),
///   - l'avancement est sauvegardé régulièrement pour pouvoir reprendre (voir Checkpoint, --resume),
///   - arrêt anticipé si la loss de vérification ne s'améliore plus (voir EarlyStoppingPatience),
///   - la loss ne compte que les caractères dont le masque vaut 1 (voir TextDataset),
///   - on peut mélanger un peu de texte général dans chaque batch (voir GeneralTextRatio),
///   - weight decay sur les matrices, et gradient clipping (voir WeightDecay, GradClip),
///   - accumulation de gradients : plusieurs petits batchs par étape (voir EffectiveBatchSize).
/// </summary>
public sealed class Trainer
{
    private readonly MiniGpt _model;
    private readonly TextDataset _dataset;
    private readonly TextDataset? _generalText;
    private readonly TrainingConfig _config;
    private readonly Device _device;
    private readonly string _phase;
    private readonly string? _lossCsvPath;
    private readonly Action<int>? _showSample;
    private readonly Checkpoint _checkpoint;

    // Meilleure loss de vérification vue (les poids correspondants sont dans le checkpoint).
    private float _bestValLoss = float.MaxValue;
    private int _bestStep;

    // Loss de vérification au premier pas de la phase (FirstStep) : celle du modèle de départ.
    private float? _startValLoss;

    // Premier pas de ce lancement (FirstStep, ou le pas sauvegardé si on reprend) : sert au temps restant.
    private int _startStep;

    // Petits batchs par étape (EffectiveBatchSize / BatchSize), et pic de mémoire GPU vu.
    private readonly int _accumulation;
    private long _peakGpuMb;

    /// <param name="dataset">Le texte principal. La loss de vérification (et donc le meilleur modèle) se mesure dessus.</param>
    /// <param name="generalText">
    /// Texte général mélangé dans les batchs d'entraînement (part = config.GeneralTextRatio),
    /// pour que le fine-tuning n'efface pas ce que le pré-entraînement a appris. null = pas de mélange.
    /// </param>
    /// <param name="phase">Nom court de la phase ("pretrain", "finetune"), écrit dans le CSV.</param>
    /// <param name="lossCsvPath">Fichier CSV où ajouter une ligne à chaque mesure de loss. null = pas de CSV.</param>
    /// <param name="showSample">Affiche une réponse d'exemple, appelé tous les config.SampleInterval pas.</param>
    /// <param name="checkpoint">Où sauvegarder l'avancement et les meilleurs poids.</param>
    public Trainer(MiniGpt model, TextDataset dataset, TrainingConfig config, Device device, Checkpoint checkpoint,
        TextDataset? generalText = null, string phase = "", string? lossCsvPath = null, Action<int>? showSample = null)
    {
        _checkpoint = checkpoint;
        _model = model;
        _dataset = dataset;
        _generalText = generalText;
        _config = config;
        _device = device;
        _phase = phase;
        _lossCsvPath = lossCsvPath;
        _showSample = showSample;

        if (config.EffectiveBatchSize % config.BatchSize != 0)
            throw new ArgumentException($"EffectiveBatchSize ({config.EffectiveBatchSize}) doit être un multiple de BatchSize ({config.BatchSize}).");
        _accumulation = config.EffectiveBatchSize / config.BatchSize;
    }

    /// <param name="resume">
    /// Reprise d'un entraînement interrompu : on repart du pas sauvegardé, avec l'état de l'optimiseur
    /// et la meilleure loss d'avant (les poids du modèle ont déjà été rechargés). null = depuis FirstStep.
    /// </param>
    /// <param name="optimizerPath">
    /// Sans reprise : état d'AdamW sauvegardé avec le modèle, à recharger (continuation, --continue).
    /// null ou fichier absent = AdamW repart de zéro.
    /// </param>
    public TrainingResult Train(CheckpointState? resume = null, string? optimizerPath = null)
    {
        var optimizer = CreateOptimizer();
        var clock = Stopwatch.StartNew();
        _model.train();
        _startStep = _config.FirstStep;

        if (resume is not null)
        {
            _checkpoint.LoadOptimizer(optimizer);
            _startStep = resume.Step;
            _startValLoss = resume.StartValLoss;
            if (_checkpoint.ReadBest() is { } best)
                (_bestValLoss, _bestStep) = (best.ValLoss, best.Step);
            Console.WriteLine($"  Reprise au pas {_startStep} (meilleure loss de vérification : {_bestValLoss:F3}, au pas {_bestStep}).");
        }
        else if (optimizerPath is not null)
        {
            if (File.Exists(optimizerPath))
            {
                optimizer.load_state_dict(optimizerPath);
                Console.WriteLine("  État de l'optimiseur (AdamW) rechargé.");
            }
            else
            {
                Console.WriteLine("  Pas d'état d'optimiseur sauvegardé avec ce modèle : AdamW repart de zéro.");
            }
        }

        Console.WriteLine($"  Batch : {_config.BatchSize} x {_accumulation} = {_config.EffectiveBatchSize} extraits " +
                          $"de {_model.Config.BlockSize} tokens par étape.");

        int lastStep = _startStep;
        for (int step = _startStep; step <= _config.Steps; step++)
        {
            lastStep = step;

            // Mesure aussi au premier pas (la loss du modèle de départ, à battre) et au dernier,
            // même s'ils ne tombent pas sur EvalInterval (une continuation peut commencer au pas 9 137).
            bool last = step == _config.Steps;
            if (step % _config.EvalInterval == 0 || step == _config.FirstStep || last)
            {
                Report(step, clock.Elapsed);

                // Arrêt anticipé : plus de progrès depuis trop longtemps, continuer ne sert à rien.
                if (_config.EarlyStoppingPatience > 0 && step - _bestStep >= _config.EarlyStoppingPatience)
                {
                    Console.WriteLine($"  Arrêt anticipé au pas {step} : pas d'amélioration de la loss de vérification " +
                                      $"depuis {step - _bestStep} pas (meilleure au pas {_bestStep}).");
                    break;
                }
            }

            // Dernier pas mesuré : un pas de plus ne serait jamais vérifié, donc jamais gardé.
            if (last) break;

            if (step % _config.CheckpointInterval == 0)
                _checkpoint.SaveProgress(_model, optimizer, new CheckpointState(_phase, step, _checkpoint.RunId, _startValLoss));

            if (_showSample is not null && step > 0 && step % _config.SampleInterval == 0)
            {
                _showSample(step);
                _model.train();   // la génération a mis le modèle en mode évaluation
            }

            // Tous les tenseurs créés dans ce scope sont libérés à la fin du tour (mémoire).
            using var scope = NewDisposeScope();

            foreach (var group in optimizer.ParamGroups)
                group.LearningRate = LearningRateAt(step);

            // Accumulation : backward() ADDITIONNE les gradients tant qu'on ne les remet pas à zéro.
            // Chaque petit batch compte pour 1/accumulation : à la fin, les gradients sont ceux d'un
            // seul gros batch de EffectiveBatchSize extraits, mais sans jamais l'avoir en mémoire d'un coup.
            optimizer.zero_grad();
            for (int micro = 0; micro < _accumulation; micro++)
            {
                // Scope par petit batch : ses activations sont libérées avant le suivant.
                using var microScope = NewDisposeScope();
                var (x, y, mask) = GetTrainingBatch();
                (ComputeLoss(x, y, mask) / _accumulation).backward();
            }

            nn.utils.clip_grad_norm_(_model.parameters(), _config.GradClip);
            optimizer.step();
        }

        // On remet les poids qui avaient la meilleure loss de vérification (optimiseur sauvegardé avant, voir SaveFinalOptimizer).
        string optimizerState = _checkpoint.SaveFinalOptimizer(optimizer);
        _checkpoint.LoadBest(_model);
        Console.WriteLine($"Meilleur modèle gardé : étape {_bestStep}, loss vérification {_bestValLoss:F3}.");

        if (_peakGpuMb > 0)
            Console.WriteLine($"Mémoire GPU utilisée au maximum : {_peakGpuMb:N0} Mo.");

        return new TrainingResult(_startValLoss ?? _bestValLoss, _bestValLoss, _bestStep, lastStep, optimizerState);
    }

    /// <summary>
    /// AdamW avec weight decay seulement sur les matrices (poids des couches, embeddings).
    /// Les biais et les LayerNorm (vecteurs, 1 dimension) n'en ont pas : les tirer vers 0 n'aide pas,
    /// et un LayerNorm à 0 éteindrait sa couche. C'est le réglage de GPT-2 / nanoGPT.
    /// </summary>
    private TorchSharp.Modules.AdamW CreateOptimizer()
    {
        var parameters = _model.parameters().ToList();
        var matrices = parameters.Where(p => p.dim() >= 2);
        var vectors = parameters.Where(p => p.dim() < 2);

        return optim.AdamW(new[]
        {
            new TorchSharp.Modules.AdamW.ParamGroup(matrices, weight_decay: _config.WeightDecay),
            new TorchSharp.Modules.AdamW.ParamGroup(vectors, weight_decay: 0),
        }, lr: _config.LearningRate);
    }

    /// <summary>
    /// Warmup puis cosinus :
    ///   - pendant WarmupSteps, la vitesse monte en ligne droite de 0 à LearningRate,
    ///   - ensuite elle descend doucement en suivant une demi-courbe cosinus jusqu'à MinLearningRate.
    /// Grands pas au début pour avancer vite, petits pas à la fin pour affiner.
    /// </summary>
    private double LearningRateAt(int step)
    {
        // Compté depuis FirstStep : une continuation a son propre warmup et son propre cosinus.
        int done = step - _config.FirstStep;
        int total = _config.Steps - _config.FirstStep;
        if (done < _config.WarmupSteps)
            return _config.LearningRate * (done + 1) / _config.WarmupSteps;

        double progress = (double)(done - _config.WarmupSteps) / Math.Max(1, total - _config.WarmupSteps);
        double cosine = 0.5 * (1 + Math.Cos(Math.PI * Math.Min(progress, 1.0)));   // 1 -> 0
        return _config.MinLearningRate + (_config.LearningRate - _config.MinLearningRate) * cosine;
    }

    /// <summary>
    /// Un (petit) batch d'entraînement. Exemple avec BatchSize 16 et GeneralTextRatio 0.2 :
    /// 13 extraits du texte principal + 3 extraits du texte général, empilés ensemble.
    /// </summary>
    private (Tensor X, Tensor Y, Tensor Mask) GetTrainingBatch()
    {
        int blockSize = _model.Config.BlockSize;
        int general = _generalText is null ? 0 : (int)Math.Round(_config.BatchSize * _config.GeneralTextRatio);

        var main = _dataset.GetBatch(validation: false, _config.BatchSize - general, blockSize, _device);
        if (general == 0)
            return main;

        var extra = _generalText!.GetBatch(validation: false, general, blockSize, _device);
        return (cat(new[] { main.X, extra.X }, 0), cat(new[] { main.Y, extra.Y }, 0), cat(new[] { main.Mask, extra.Mask }, 0));
    }

    /// <summary>
    /// Cross-entropy moyenne, mais seulement sur les caractères dont le masque vaut 1.
    /// Masque partout à 1 = la cross-entropy classique.
    /// </summary>
    private Tensor ComputeLoss(Tensor x, Tensor y, Tensor mask)
    {
        var logits = _model.forward(x);                       // (batch, temps, vocab)
        int vocab = _model.Config.VocabSize;

        // Une erreur par caractère (pas encore de moyenne), puis on éteint ceux qui ne comptent pas.
        var perChar = functional.cross_entropy(logits.view(-1, vocab), y.view(-1), reduction: Reduction.None);
        var weights = mask.view(-1);

        // clamp : un extrait peut tomber entièrement dans des lignes "Toi:" (aucun caractère noté).
        return (perChar * weights).sum() / weights.sum().clamp_min(1);
    }

    private void Report(int step, TimeSpan elapsed)
    {
        float train = EstimateLoss(validation: false);
        float val = EstimateLoss(validation: true);
        _startValLoss ??= val;

        string best = "";
        if (val < _bestValLoss)
        {
            (_bestValLoss, _bestStep) = (val, step);
            _checkpoint.SaveBest(_model, new BestScore(val, step));
            best = "  <- meilleur";
        }

        Console.WriteLine($"  Étape {step,5}/{_config.Steps}  |  loss entraînement {train:F3}  |  loss vérification {val:F3}" +
                          $"  |  lr {LearningRateAt(step):0.0e0}{GpuMemoryText()}  |  {Format(elapsed)}  |  reste {Remaining(step, elapsed)}{best}");

        WriteCsvLine(step, train, val, elapsed);
    }

    /// <summary>"  |  GPU 2 345/4 096 Mo" sur GPU, rien sur CPU. Retient aussi le pic.</summary>
    private string GpuMemoryText()
    {
        if (_device.type != DeviceType.CUDA || GpuMemory.Read() is not var (used, total))
            return "";

        _peakGpuMb = Math.Max(_peakGpuMb, used);
        return $"  |  GPU {used:N0}/{total:N0} Mo";
    }

    /// <summary>
    /// Ajoute une ligne au CSV des loss (s'ouvre dans Excel, ou se trace en courbe).
    /// Nombres avec un point (format international), sinon la virgule française casserait les colonnes.
    /// </summary>
    private void WriteCsvLine(int step, float train, float val, TimeSpan elapsed)
    {
        if (_lossCsvPath is null) return;

        if (!File.Exists(_lossCsvPath))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_lossCsvPath)!);
            File.WriteAllText(_lossCsvPath, "phase,etape,loss_entrainement,loss_verification,learning_rate,secondes\n");
        }

        File.AppendAllText(_lossCsvPath, string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"{_phase},{step},{train:F4},{val:F4},{LearningRateAt(step):G4},{elapsed.TotalSeconds:F0}\n"));
    }

    /// <summary>Moyenne de la loss sur plusieurs batchs, sans apprendre (mode évaluation).</summary>
    private float EstimateLoss(bool validation)
    {
        _model.eval();
        using var noGrad = no_grad();

        float total = 0;
        for (int i = 0; i < _config.EvalBatches; i++)
        {
            // Un scope par batch : TorchSharp ne libère un tenseur qu'à la fin de son scope. Avec un seul
            // scope pour toute la boucle, les activations des EvalBatches passes s'empilaient sur le GPU.
            using var scope = NewDisposeScope();
            var (x, y, mask) = _dataset.GetBatch(validation, _config.BatchSize, _model.Config.BlockSize, _device);
            total += ComputeLoss(x, y, mask).item<float>();
        }

        _model.train();
        return total / _config.EvalBatches;
    }

    /// <summary>Temps restant estimé : durée moyenne d'une étape depuis le lancement x étapes restantes.</summary>
    private string Remaining(int step, TimeSpan elapsed)
    {
        if (step == _startStep) return "--:--";
        return Format(elapsed / (step - _startStep) * (_config.Steps - step));
    }

    /// <summary>"mm:ss", ou "h:mm:ss" à partir d'une heure (un entraînement long peut durer des heures).</summary>
    private static string Format(TimeSpan time) =>
        time.TotalHours >= 1
            ? $"{(int)time.TotalHours}:{time:mm\\:ss}"
            : time.ToString(@"mm\:ss");
}
