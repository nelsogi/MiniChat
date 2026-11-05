using System.Diagnostics;
using MiniChat.Config;
using MiniChat.Data;
using MiniChat.Model;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Training;

/// <summary>
/// La boucle d'apprentissage. À chaque étape :
///   1. le modèle prédit le caractère suivant pour chaque position,
///   2. on mesure son erreur (la "loss", cross-entropy),
///   3. backward() calcule comment chaque poids a contribué à l'erreur,
///   4. l'optimiseur AdamW ajuste un peu chaque poids pour réduire l'erreur.
///
/// En plus :
///   - la vitesse d'apprentissage suit un warmup puis une descente en cosinus,
///   - on garde une copie des poids quand la loss de vérification est la plus basse,
///     et on la remet dans le modèle à la fin (le modèle sauvegardé = le meilleur vu),
///   - la loss ne compte que les caractères dont le masque vaut 1 (voir TextDataset),
///   - on peut mélanger un peu de texte général dans chaque batch (voir GeneralTextRatio),
///   - weight decay sur les matrices, et gradient clipping (voir WeightDecay, GradClip).
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

    // Meilleurs poids rencontrés (copies, indépendantes du modèle qui continue d'apprendre).
    private Dictionary<string, Tensor>? _bestWeights;
    private float _bestValLoss = float.MaxValue;
    private int _bestStep;

    /// <param name="dataset">Le texte principal. La loss de vérification (et donc le meilleur modèle) se mesure dessus.</param>
    /// <param name="generalText">
    /// Texte général mélangé dans les batchs d'entraînement (part = config.GeneralTextRatio),
    /// pour que le fine-tuning n'efface pas ce que le pré-entraînement a appris. null = pas de mélange.
    /// </param>
    /// <param name="phase">Nom court de la phase ("pretrain", "finetune"), écrit dans le CSV.</param>
    /// <param name="lossCsvPath">Fichier CSV où ajouter une ligne à chaque mesure de loss. null = pas de CSV.</param>
    /// <param name="showSample">Affiche une réponse d'exemple, appelé tous les config.SampleInterval pas.</param>
    public Trainer(MiniGpt model, TextDataset dataset, TrainingConfig config, Device device, TextDataset? generalText = null,
        string phase = "", string? lossCsvPath = null, Action<int>? showSample = null)
    {
        _model = model;
        _dataset = dataset;
        _generalText = generalText;
        _config = config;
        _device = device;
        _phase = phase;
        _lossCsvPath = lossCsvPath;
        _showSample = showSample;
    }

    public void Train()
    {
        var optimizer = CreateOptimizer();
        var clock = Stopwatch.StartNew();
        _model.train();

        for (int step = 0; step <= _config.Steps; step++)
        {
            if (step % _config.EvalInterval == 0)
                Report(step, clock.Elapsed);

            if (_showSample is not null && step > 0 && step % _config.SampleInterval == 0)
            {
                _showSample(step);
                _model.train();   // la génération a mis le modèle en mode évaluation
            }

            // Tous les tenseurs créés dans ce scope sont libérés à la fin du tour (mémoire).
            using var scope = NewDisposeScope();

            foreach (var group in optimizer.ParamGroups)
                group.LearningRate = LearningRateAt(step);

            var (x, y, mask) = GetTrainingBatch();
            var loss = ComputeLoss(x, y, mask);

            optimizer.zero_grad();
            loss.backward();
            nn.utils.clip_grad_norm_(_model.parameters(), _config.GradClip);
            optimizer.step();
        }

        RestoreBestWeights();
    }

    /// <summary>
    /// AdamW avec weight decay seulement sur les matrices (poids des couches, embeddings).
    /// Les biais et les LayerNorm (vecteurs, 1 dimension) n'en ont pas : les tirer vers 0 n'aide pas,
    /// et un LayerNorm à 0 éteindrait sa couche. C'est le réglage de GPT-2 / nanoGPT.
    /// </summary>
    private optim.Optimizer CreateOptimizer()
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
        if (step < _config.WarmupSteps)
            return _config.LearningRate * (step + 1) / _config.WarmupSteps;

        double progress = (double)(step - _config.WarmupSteps) / Math.Max(1, _config.Steps - _config.WarmupSteps);
        double cosine = 0.5 * (1 + Math.Cos(Math.PI * Math.Min(progress, 1.0)));   // 1 -> 0
        return _config.MinLearningRate + (_config.LearningRate - _config.MinLearningRate) * cosine;
    }

    /// <summary>
    /// Un batch d'entraînement. Exemple avec BatchSize 32 et GeneralTextRatio 0.2 :
    /// 26 extraits du texte principal + 6 extraits du texte général, empilés ensemble.
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

        string best = "";
        if (val < _bestValLoss)
        {
            SaveBestWeights(step, val);
            best = "  <- meilleur";
        }

        Console.WriteLine($"  Étape {step,5}/{_config.Steps}  |  loss entraînement {train:F3}  |  loss vérification {val:F3}" +
                          $"  |  lr {LearningRateAt(step):0.0e0}  |  {Format(elapsed)}  |  reste {Remaining(step, elapsed)}{best}");

        WriteCsvLine(step, train, val, elapsed);
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
        using var scope = NewDisposeScope();

        float total = 0;
        for (int i = 0; i < _config.EvalBatches; i++)
        {
            var (x, y, mask) = _dataset.GetBatch(validation, _config.BatchSize, _model.Config.BlockSize, _device);
            total += ComputeLoss(x, y, mask).item<float>();
        }

        _model.train();
        return total / _config.EvalBatches;
    }

    /// <summary>Temps restant estimé : durée moyenne d'une étape jusqu'ici x étapes restantes.</summary>
    private string Remaining(int step, TimeSpan elapsed)
    {
        if (step == 0) return "--:--";
        return Format(elapsed / step * (_config.Steps - step));
    }

    /// <summary>"mm:ss", ou "h:mm:ss" à partir d'une heure (un entraînement long peut durer des heures).</summary>
    private static string Format(TimeSpan time) =>
        time.TotalHours >= 1
            ? $"{(int)time.TotalHours}:{time:mm\\:ss}"
            : time.ToString(@"mm\:ss");

    /// <summary>Copie les poids actuels (sans calcul de gradient, hors de tout DisposeScope).</summary>
    private void SaveBestWeights(int step, float valLoss)
    {
        using var noGrad = no_grad();

        if (_bestWeights is not null)
            foreach (var tensor in _bestWeights.Values) tensor.Dispose();

        _bestWeights = _model.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        _bestValLoss = valLoss;
        _bestStep = step;
    }

    /// <summary>Remet dans le modèle les poids qui avaient la loss de vérification la plus basse.</summary>
    private void RestoreBestWeights()
    {
        if (_bestWeights is null) return;

        _model.load_state_dict(_bestWeights);
        foreach (var tensor in _bestWeights.Values) tensor.Dispose();
        _bestWeights = null;

        Console.WriteLine($"Meilleur modèle gardé : étape {_bestStep}, loss vérification {_bestValLoss:F3}.");
    }
}
