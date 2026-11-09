using System.Text.Json;
using MiniChat.Model;
using MiniChat.Storage;
using MiniChat.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace MiniChat.Training;

/// <summary>
/// Où en est un entraînement. Step = prochain pas à faire (les pas d'avant sont faits).
/// StartValLoss = loss de vérification du modèle au premier pas de la phase (null si pas encore mesurée) :
/// pour une continuation, c'est la loss de l'ancien modèle, à battre pour le remplacer.
/// Steps = dernier pas prévu de la phase (calculé depuis --epochs) : la reprise garde la même fin, donc le même
/// cosinus. null dans les sauvegardes d'avant : recalculé depuis la taille des données.
/// </summary>
public sealed record CheckpointState(string Phase, int Step, string RunId, float? StartValLoss = null, int? Steps = null);

/// <summary>
/// Une continuation (--continue, --continue-finetune) : Steps pas de plus à partir de FirstStep,
/// à la vitesse LearningRate. Gardée pour que --resume refasse les mêmes réglages.
/// </summary>
public sealed record ContinueState(string Phase, int FirstStep, int Steps, double LearningRate);

/// <summary>La meilleure loss de vérification de la phase en cours, et à quel pas (va avec best.dat).</summary>
public sealed record BestScore(float ValLoss, int Step);

/// <summary>
/// Sauvegarde régulière d'un entraînement en cours, pour pouvoir le reprendre (--resume)
/// après une coupure (plantage, PC éteint, Ctrl+C). Dossier "checkpoint" :
///   config.json, tokenizer.json, weights.dat   -> le modèle tel qu'il est maintenant (via ModelStore)
///   optimizer.dat                              -> l'état d'AdamW (ses moyennes de gradients)
///   best.dat, best.json                        -> les poids qui ont eu la meilleure loss de vérification,
///                                                 et cette loss (écrits ensemble, à chaque amélioration)
///   state.json                                 -> phase et pas de la dernière sauvegarde (voir CheckpointState)
///   continue.json                              -> seulement pour --continue : ses réglages (voir ContinueState)
/// Supprimé à la fin d'un entraînement complet : s'il existe, c'est qu'un entraînement a été interrompu.
/// </summary>
public sealed class Checkpoint
{
    private readonly string _folder;
    private readonly ModelStore _store;

    public Checkpoint(string folder, string runId)
    {
        _folder = folder;
        _store = new ModelStore(folder);
        RunId = runId;
    }

    /// <summary>Date de l'entraînement : à la reprise, on continue les mêmes fichiers results/.</summary>
    public string RunId { get; }

    private string StatePath => Path.Combine(_folder, "state.json");
    private string OptimizerPath => Path.Combine(_folder, "optimizer.dat");
    private string BestPath => Path.Combine(_folder, "best.dat");
    private string BestScorePath => Path.Combine(_folder, "best.json");
    private string ContinuePath => Path.Combine(_folder, "continue.json");

    public static bool ExistsIn(string folder) =>
        File.Exists(Path.Combine(folder, "state.json")) && new ModelStore(folder).Exists;

    public static CheckpointState ReadState(string folder) =>
        JsonSerializer.Deserialize<CheckpointState>(File.ReadAllText(Path.Combine(folder, "state.json")))
        ?? throw new InvalidDataException("checkpoint/state.json illisible.");

    /// <summary>
    /// Nouveau départ : efface l'ancienne sauvegarde et enregistre la forme du modèle et le tokenizer.
    /// continuation : réglages d'un --continue, relus par ReadContinuation à la reprise.
    /// </summary>
    public void Start(MiniGpt model, BpeTokenizer tokenizer, ContinueState? continuation = null)
    {
        Delete();
        _store.Save(model, tokenizer);
        if (continuation is not null)
            File.WriteAllText(ContinuePath, JsonSerializer.Serialize(continuation));
    }

    /// <summary>Les réglages du --continue en cours, ou null si c'est un entraînement normal.</summary>
    public ContinueState? ReadContinuation() =>
        File.Exists(ContinuePath) ? JsonSerializer.Deserialize<ContinueState>(File.ReadAllText(ContinuePath)) : null;

    /// <summary>Le modèle (poids au dernier checkpoint) et son tokenizer.</summary>
    public (MiniGpt Model, BpeTokenizer Tokenizer) LoadModel(Device device) => _store.Load(device);

    /// <summary>
    /// Sauvegarde l'avancement. Tout est écrit dans des .tmp puis renommé, state.json en dernier :
    /// une coupure PENDANT la sauvegarde laisse l'ancienne intacte.
    /// </summary>
    public void SaveProgress(MiniGpt model, TorchSharp.Modules.OptimizerHelper optimizer, CheckpointState state)
    {
        // ponytail: une coupure entre deux renommages (quelques ms) peut mélanger ancien et nouveau ;
        // un dossier par sauvegarde + un pointeur réglerait ça si ça arrive un jour.
        model.save(_store.WeightsPath + ".tmp");
        optimizer.save_state_dict(OptimizerPath + ".tmp");
        File.WriteAllText(StatePath + ".tmp", JsonSerializer.Serialize(state));

        File.Move(_store.WeightsPath + ".tmp", _store.WeightsPath, overwrite: true);
        File.Move(OptimizerPath + ".tmp", OptimizerPath, overwrite: true);
        File.Move(StatePath + ".tmp", StatePath, overwrite: true);
    }

    public void LoadOptimizer(TorchSharp.Modules.OptimizerHelper optimizer) => optimizer.load_state_dict(OptimizerPath);

    /// <summary>
    /// État d'AdamW à la fin de la phase, dans final-optimizer.dat (copié ensuite avec le modèle, voir ModelStore).
    /// À appeler AVANT LoadBest : recharger des poids remplace les tenseurs des paramètres,
    /// et l'optimiseur, qui pointe encore sur les anciens, ne peut plus se sauvegarder.
    /// </summary>
    public string SaveFinalOptimizer(TorchSharp.Modules.OptimizerHelper optimizer)
    {
        string path = Path.Combine(_folder, "final-optimizer.dat");
        optimizer.save_state_dict(path);
        return path;
    }

    /// <summary>
    /// Garde les poids actuels comme "meilleurs" (sur disque : ils survivent à une coupure).
    /// La loss est écrite à côté, au même moment : à la reprise, poids et score correspondent toujours.
    /// </summary>
    public void SaveBest(MiniGpt model, BestScore score)
    {
        model.save(BestPath + ".tmp");
        File.WriteAllText(BestScorePath + ".tmp", JsonSerializer.Serialize(score));
        File.Move(BestPath + ".tmp", BestPath, overwrite: true);
        File.Move(BestScorePath + ".tmp", BestScorePath, overwrite: true);
    }

    /// <summary>Le meilleur score de la phase en cours, ou null s'il n'y en a pas encore.</summary>
    public BestScore? ReadBest() =>
        File.Exists(BestScorePath) ? JsonSerializer.Deserialize<BestScore>(File.ReadAllText(BestScorePath)) : null;

    /// <summary>Remet les meilleurs poids dans le modèle.</summary>
    public void LoadBest(MiniGpt model)
    {
        if (File.Exists(BestPath))
            model.load(BestPath);
    }

    public void Delete()
    {
        if (Directory.Exists(_folder))
            Directory.Delete(_folder, recursive: true);
    }
}
