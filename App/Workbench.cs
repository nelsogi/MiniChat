using System.Text;
using MiniChat.Chat;
using MiniChat.Config;
using MiniChat.Data;
using MiniChat.Evaluation;
using MiniChat.Generation;
using MiniChat.Model;
using MiniChat.Storage;
using MiniChat.Tokenization;
using MiniChat.Training;
using MiniChat.UI;
using Spectre.Console;
using TorchSharp;
using static TorchSharp.torch;

namespace MiniChat.App;

/// <summary>Une erreur de l'utilisateur (rien à reprendre, pas de modèle...) : on affiche le message, sans plantage.</summary>
public sealed class UserException(string message) : Exception(message);

/// <summary>
/// Le modèle chargé et tout ce qui le change : entraîner (/train, --train), refaire le fine-tuning (/finetune),
/// continuer (/continue), reprendre (/resume), et passer les questions de test (/test).
/// Les commandes de Nel Code comme les options en ligne de commande passent par ici.
///
/// Deux dossiers :
///   data/    texte.txt, dialogues.txt, questions-test.txt
///   models/  model/, model-pretrain/, et ce que l'entraînement écrit : checkpoint/, results/
/// </summary>
public sealed class Workbench
{
    public const string ChatModel = "model";
    public const string PretrainedModel = "model-pretrain";

    // Taille du vocabulaire BPE : plus grand = mots entiers plus souvent, mais plus de données nécessaires.
    // 4000 : texte.txt fait maintenant des centaines de millions de tokens, de quoi apprendre chaque token.
    private const int TokenizerVocabSize = 4000;

    // Le tokenizer apprend sur un échantillon de texte.txt réparti dans tout le fichier, pas sur le fichier entier :
    // ~1 min 30 pour 20 Mo et 4000 tokens, et les fusions apprises ne changent presque plus au-delà.
    private const int TokenizerSampleMb = 20;

    private readonly string _pretrainPath;
    private readonly string _dialoguesPath;
    private readonly string _questionsPath;
    private readonly string _checkpointFolder;   // sauvegarde régulière de l'entraînement en cours (pour /resume)
    private readonly string _resultsFolder;
    private readonly string _cacheFolder;        // tokens de texte.txt et dialogues.txt déjà découpés (voir TokenCache)

    // Deux sauvegardes : le modèle pré-entraîné (base pour refaire le fine-tuning) et le modèle final.
    private readonly ModelStore _pretrainStore;
    private readonly ModelStore _store;

    // L'arrêt demandé pour l'entraînement en cours (voir Training), vide le reste du temps.
    private CancellationToken _cancellation;

    // Le tokenizer de Model (pour /save).
    private BpeTokenizer? _tokenizer;

    // Les modèles gardés avec /save : models/saved/<nom>/, même contenu que model/.
    private string SavedFolder => Path.Combine(ModelsFolder, "saved");

    public Workbench(string dataFolder, string modelsFolder, Device device)
    {
        DataFolder = dataFolder;
        ModelsFolder = modelsFolder;
        Device = device;
        _pretrainPath = Path.Combine(dataFolder, "texte.txt");
        _dialoguesPath = Path.Combine(dataFolder, "dialogues.txt");
        _questionsPath = Path.Combine(dataFolder, "questions-test.txt");
        _checkpointFolder = Path.Combine(modelsFolder, "checkpoint");
        _resultsFolder = Path.Combine(modelsFolder, "results");
        _cacheFolder = Path.Combine(modelsFolder, "cache");
        _pretrainStore = new ModelStore(Path.Combine(modelsFolder, PretrainedModel));
        _store = new ModelStore(Path.Combine(modelsFolder, ChatModel));
    }

    public string DataFolder { get; }
    public string ModelsFolder { get; }

    /// <summary>GPU NVIDIA s'il y en a un d'utilisable, sinon CPU. Le modèle et les tenseurs suivent ce device.</summary>
    public Device Device { get; }
    public bool OnGpu => Device.type == DeviceType.CUDA;

    /// <summary>Le modèle en mémoire, null s'il n'y en a pas (jamais entraîné, ou pendant un entraînement).</summary>
    public MiniGpt? Model { get; private set; }

    /// <summary>Ce qui écrit avec Model ; change à chaque nouveau modèle.</summary>
    public TextGenerator? Generator { get; private set; }

    /// <summary>D'où vient Model : ChatModel, ou PretrainedModel (il ne sait pas encore discuter).</summary>
    public string? ModelSource { get; private set; }

    public bool HasInterruptedTraining => Checkpoint.ExistsIn(_checkpointFolder);

    /// <summary>Charge le modèle par défaut (DefaultModel) s'il existe. Sinon, pas de modèle : Nel Code propose /train.</summary>
    public void LoadSaved()
    {
        var (store, name) = DefaultModel == PretrainedModel && _pretrainStore.Exists
            ? (_pretrainStore, PretrainedModel)
            : (_store, ChatModel);
        if (!store.Exists)
            return;

        // Un modèle qu'on ne peut plus lire (format d'avant) ne doit pas empêcher Nel Code de démarrer.
        try
        {
            SetModel(store.Load(Device), name);
        }
        catch (InvalidDataException e)
        {
            Messages.Warning(e.Message);
        }
    }

    /// <summary>
    /// Le modèle que LoadSaved charge : ChatModel, ou PretrainedModel (pour voir ce que donne le pré-entraînement seul).
    /// Vient de settings.json. PretrainedModel absent : on charge ChatModel.
    /// </summary>
    public string DefaultModel { get; set; } = ChatModel;

    // Les méthodes qui entraînent prennent un CancellationToken (Échap dans Nel Code) : la phase en cours
    // s'arrête proprement (meilleurs poids sauvegardés, avancement gardé pour /resume) et la suivante n'est pas lancée.

    /// <summary>Nouveau modèle : pré-entraînement puis fine-tuning.</summary>
    public void Train(CancellationToken cancellation = default) => Training(cancellation, () =>
    {
        WarnIfInterrupted();
        string now = Now();
        var (log, checkpoint, answersPath) = NewRun(now, now);

        var (model, tokenizer) = CreateModel();
        checkpoint.Start(model, tokenizer);
        Pretrain(model, tokenizer, _pretrainStore, TrainingConfig.Pretraining, log, checkpoint, resume: null);
        if (Stopped(model, tokenizer, pretrainOnly: true)) return;
        FineTune(model, tokenizer, _store, TrainingConfig.FineTuning, log, checkpoint, resume: null);
        Finish(model, tokenizer, "pré-entraînement + fine-tuning", log, checkpoint, answersPath, pretrainOnly: false);
    });

    /// <summary>Refait seulement la phase 2, en repartant du modèle pré-entraîné (pas du modèle final).</summary>
    public void FineTuneOnly(CancellationToken cancellation = default) => Training(cancellation, () =>
    {
        if (!_pretrainStore.Exists)
            throw new UserException($"Pas de modèle pré-entraîné dans '{PretrainedModel}' : lance d'abord /train.");

        WarnIfInterrupted();
        string now = Now();
        var (log, checkpoint, answersPath) = NewRun(now, now);

        var (model, tokenizer) = _pretrainStore.Load(Device);
        Messages.Info($"Modèle pré-entraîné chargé depuis '{PretrainedModel}'.");
        checkpoint.Start(model, tokenizer);
        FineTune(model, tokenizer, _store, TrainingConfig.FineTuning, log, checkpoint, resume: null);
        Finish(model, tokenizer, "fine-tuning seul (--finetune)", log, checkpoint, answersPath, pretrainOnly: false);
    });

    /// <summary>
    /// steps pas de plus sur un modèle déjà entraîné, avec les données actuelles de data/ (on peut en avoir ajouté).
    /// phase "pretrain" : depuis model-pretrain/ ; "finetune" : depuis model/.
    /// learningRate null : vitesse par défaut (voir TrainingConfig.Continuation).
    /// </summary>
    public void Continue(string phase, int steps, double? learningRate, CancellationToken cancellation = default) =>
        Training(cancellation, () =>
    {
        var store = StoreFor(phase);
        if (!store.Exists)
            throw new UserException($"Rien à continuer : pas de modèle dans '{StoreName(phase)}'. Lance d'abord /train.");

        WarnIfInterrupted();
        // Le CSV des loss continue celui de l'entraînement d'origine (date gardée dans training.json),
        // mais les réponses aux questions ont leur propre fichier.
        var info = store.ReadInfo();
        string now = Now();
        var (log, checkpoint, answersPath) = NewRun(info?.RunId ?? now, now);

        // Repart du modèle sauvegardé, avec SON tokenizer : jamais de nouveau BPE, sinon les ids ne voudraient
        // plus rien dire pour le modèle. Les caractères des nouvelles données qu'il ne connaît pas sont ignorés.
        var (model, tokenizer) = store.Load(Device);
        int firstStep = info?.Step ?? 0;
        var continuation = new ContinueState(phase, firstStep, steps,
            BaseConfig(phase).Continuation(firstStep, steps, learningRate).LearningRate);

        Messages.Info($"Modèle chargé depuis '{StoreName(phase)}'" +
                      (info is null ? " (sauvegardé avant --continue : pas numérotés depuis 0)." : $", arrêté au pas {firstStep}."));
        checkpoint.Start(model, tokenizer, continuation);
        ContinuePhase(continuation, model, tokenizer, store, info, log, checkpoint, resume: null);
        Finish(model, tokenizer, phase == "pretrain"
                ? $"pré-entraînement continué de {steps} pas (--continue)"
                : $"fine-tuning continué de {steps} pas (--continue-finetune)",
            log, checkpoint, answersPath, pretrainOnly: phase == "pretrain");
    });

    /// <summary>Reprend la phase en cours là où elle s'est arrêtée (poids, optimiseur, pas, meilleure loss).</summary>
    public void Resume(CancellationToken cancellation = default) => Training(cancellation, () =>
    {
        if (!Checkpoint.ExistsIn(_checkpointFolder))
            throw new UserException("Rien à reprendre : pas d'entraînement interrompu dans 'checkpoint'.");

        // Une reprise garde la date de l'entraînement d'origine : le CSV continue au même endroit.
        var state = Checkpoint.ReadState(_checkpointFolder);
        var continuation = new Checkpoint(_checkpointFolder, state.RunId).ReadContinuation();
        var (log, checkpoint, answersPath) = NewRun(state.RunId, continuation is null ? state.RunId : Now());

        var (model, tokenizer) = checkpoint.LoadModel(Device);
        Messages.Info($"Reprise de l'entraînement du {state.RunId} : phase '{state.Phase}', pas {state.Step}.");

        if (continuation is not null)
        {
            // Un --continue interrompu : mêmes réglages, et toujours rien qu'une phase.
            var store = StoreFor(continuation.Phase);
            ContinuePhase(continuation, model, tokenizer, store, store.ReadInfo(), log, checkpoint, state);
        }
        else if (state.Phase == "pretrain")
        {
            Pretrain(model, tokenizer, _pretrainStore, TrainingConfig.Pretraining, log, checkpoint, state);
            if (Stopped(model, tokenizer, pretrainOnly: true)) return;
            FineTune(model, tokenizer, _store, TrainingConfig.FineTuning, log, checkpoint, resume: null);
        }
        else
        {
            FineTune(model, tokenizer, _store, TrainingConfig.FineTuning, log, checkpoint, state);
        }
        Finish(model, tokenizer, "entraînement repris (--resume)", log, checkpoint, answersPath,
            pretrainOnly: continuation?.Phase == "pretrain");
    });

    /// <summary>Les questions de test sur le modèle chargé, sans rien entraîner. Réponses dans results/&lt;date&gt;-test.txt.</summary>
    public void RunTest(CancellationToken cancellation = default)
    {
        if (Model is not { } model || Generator is not { } generator)
            throw new UserException("Aucun modèle chargé : lance /train d'abord.");

        string path = Path.Combine(_resultsFolder, $"{Now()}-test.txt");
        string report = Messages.Spin("Questions de test... (Échap pour arrêter)", () => QuestionTest.Run(generator,
            _questionsPath, path, $"Test (/test) | modèle '{ModelSource}' | {model.ParameterCount:N0} paramètres", cancellation));
        ShowReport(report, path);
    }

    /// <summary>
    /// Le CSV des loss du dernier entraînement : le dernier écrit (une continuation ou une reprise
    /// écrit dans le CSV de l'entraînement d'origine). null s'il n'y en a pas encore.
    /// </summary>
    public string? LatestLossCsv() =>
        Directory.Exists(_resultsFolder)
            ? Directory.EnumerateFiles(_resultsFolder, "*-loss.csv").MaxBy(File.GetLastWriteTimeUtc)
            : null;

    /// <summary>
    /// Tous les modèles sauvegardés : model/, model-pretrain/, puis ceux gardés avec /save (saved/&lt;nom&gt;/),
    /// par ordre alphabétique.
    /// </summary>
    public IReadOnlyList<SavedModel> ListModels()
    {
        var stores = new List<(string Name, ModelStore Store)> { (ChatModel, _store), (PretrainedModel, _pretrainStore) };
        if (Directory.Exists(SavedFolder))
            stores.AddRange(Directory.EnumerateDirectories(SavedFolder)
                .Select(folder => (Path.GetFileName(folder), new ModelStore(folder)))
                .OrderBy(s => s.Item1, StringComparer.OrdinalIgnoreCase));

        return stores
            .Where(s => s.Store.Exists)
            // Les poids sont en float32 : 4 octets par paramètre (plus un en-tête de quelques octets).
            .Select(s => new SavedModel(s.Name, s.Store.ReadConfig(), s.Store.WeightsBytes / 4, s.Store.WeightsBytes,
                s.Store.SavedAt, s.Store.ReadInfo()))
            .ToList();
    }

    /// <summary>
    /// Garde le modèle chargé sous un nom (saved/&lt;nom&gt;/), pour le recharger ou le comparer plus tard,
    /// même après un nouvel entraînement. Un modèle du même nom est remplacé ; renvoie true dans ce cas.
    /// </summary>
    public bool SaveAs(string name)
    {
        if (Model is not { } model || _tokenizer is not { } tokenizer)
            throw new UserException("Aucun modèle chargé : rien à sauvegarder.");
        CheckSavedName(name);

        // L'avancement (pas, loss, date de l'entraînement) suit le modèle ; l'état d'AdamW (deux fois la taille
        // du modèle) non : un modèle gardé sert à discuter et à comparer, pas à être continué.
        var info = ModelSource is { } source && TryStore(source) is { Exists: true } from ? from.ReadInfo() : null;
        string folder = Path.Combine(SavedFolder, name);
        bool replaced = Directory.Exists(folder);
        if (replaced)
            Directory.Delete(folder, recursive: true);
        new ModelStore(folder).Save(model, tokenizer, info: info);
        return replaced;
    }

    /// <summary>Charge un modèle sauvegardé (model, model-pretrain, ou un nom de /save) : il devient celui du chat.</summary>
    public void Load(string name)
    {
        var store = RequireStore(name);
        SetModel(null, null);   // libère la mémoire (GPU) de l'ancien avant de charger le nouveau
        try
        {
            SetModel(store.Load(Device), NameOf(name));
        }
        finally
        {
            if (Model is null)
                LoadSaved();
        }
    }

    /// <summary>
    /// Les questions de test posées à deux modèles sauvegardés, dans les conditions de /test (température
    /// et graine fixes) : seule la différence entre les modèles compte. cancellation : s'arrête avant la question suivante.
    /// </summary>
    public IReadOnlyList<(string Question, string First, string Second)> Compare(string first, string second,
        CancellationToken cancellation = default)
    {
        var (firstStore, secondStore) = (RequireStore(first), RequireStore(second));
        var questions = QuestionTest.ReadQuestions(_questionsPath);
        var answers = new List<(string, string, string)>();

        var (modelA, tokenizerA) = Messages.Spin($"Chargement de '{first}'...", () => firstStore.Load(Device));
        try
        {
            var (modelB, tokenizerB) = Messages.Spin($"Chargement de '{second}'...", () => secondStore.Load(Device));
            try
            {
                var (a, b) = (new TextGenerator(modelA, tokenizerA, Device), new TextGenerator(modelB, tokenizerB, Device));
                for (int i = 0; i < questions.Length && !cancellation.IsCancellationRequested; i++)
                {
                    string question = questions[i];
                    answers.Add(Messages.Spin($"Question {i + 1}/{questions.Length}... (Échap pour arrêter)",
                        () => (question, QuestionTest.Answer(a, question), QuestionTest.Answer(b, question))));
                }
            }
            finally
            {
                modelB.Dispose();
            }
        }
        finally
        {
            modelA.Dispose();
        }
        return answers;
    }

    private ModelStore? TryStore(string name) =>
        name.Equals(ChatModel, StringComparison.OrdinalIgnoreCase) ? _store
        : name.Equals(PretrainedModel, StringComparison.OrdinalIgnoreCase) ? _pretrainStore
        : IsValidName(name) ? new ModelStore(Path.Combine(SavedFolder, name))
        : null;

    private ModelStore RequireStore(string name) =>
        TryStore(name) is { Exists: true } store
            ? store
            : throw new UserException($"Pas de modèle « {name} ». Modèles : {string.Join(", ", ListModels().Select(m => m.Name))}" +
                                      " (/models pour le détail).");

    // Le nom tel qu'il est écrit sur le disque : "/load MODEL" charge "model".
    private string NameOf(string name) =>
        ListModels().FirstOrDefault(m => m.Name.Equals(name, StringComparison.OrdinalIgnoreCase))?.Name ?? name;

    private static bool IsValidName(string name) =>
        name.Length is > 0 and <= 64 && name.All(c => char.IsLetterOrDigit(c) || c is '-' or '_' or '.') && name[0] != '.';

    private static void CheckSavedName(string name)
    {
        if (!IsValidName(name))
            throw new UserException($"Nom invalide : « {name} ». Lettres, chiffres, - _ . (64 caractères au plus).");
        if (name.Equals(ChatModel, StringComparison.OrdinalIgnoreCase) || name.Equals(PretrainedModel, StringComparison.OrdinalIgnoreCase))
            throw new UserException($"« {name} » est réservé à l'entraînement : choisis un autre nom.");
    }

    // ---------------------------------------------------------------------------

    /// <summary>
    /// Entraîne avec la mémoire (GPU surtout) libérée du modèle d'avant. Si l'entraînement échoue,
    /// on recharge le modèle sauvegardé : le chat a de nouveau quelqu'un pour répondre.
    /// </summary>
    private void Training(CancellationToken cancellation, Action train)
    {
        SetModel(null, null);
        _cancellation = cancellation;
        try
        {
            train();
        }
        finally
        {
            _cancellation = default;
            if (Model is null)
                LoadSaved();
        }
    }

    /// <summary>
    /// Arrêt demandé : la phase arrêtée a déjà sauvegardé ses meilleurs poids (ou gardé l'ancien modèle s'ils
    /// n'étaient pas meilleurs). Le checkpoint reste pour /resume ; pas de questions de test.
    /// </summary>
    private bool Stopped(MiniGpt model, BpeTokenizer tokenizer, bool pretrainOnly)
    {
        if (!_cancellation.IsCancellationRequested)
            return false;

        Messages.Warning("Entraînement arrêté (Échap). /resume pour le reprendre là où il s'est arrêté.");
        Adopt(model, tokenizer, pretrainOnly);
        return true;
    }

    /// <summary>
    /// Le modèle qui vient d'être entraîné devient celui du chat. Sauf un modèle seulement pré-entraîné :
    /// il ne sait pas encore discuter, donc on garde le modèle final s'il y en a un.
    /// </summary>
    private void Adopt(MiniGpt model, BpeTokenizer tokenizer, bool pretrainOnly)
    {
        if (pretrainOnly && _store.Exists)
        {
            model.Dispose();
            SetModel(_store.Load(Device), ChatModel);
            return;
        }
        SetModel((model, tokenizer), pretrainOnly ? PretrainedModel : ChatModel);
    }

    private void SetModel((MiniGpt Model, BpeTokenizer Tokenizer)? loaded, string? source)
    {
        if (Model is not null && !ReferenceEquals(Model, loaded?.Model))
            Model.Dispose();

        Model = loaded?.Model;
        _tokenizer = loaded?.Tokenizer;
        Generator = loaded is var (model, tokenizer) ? new TextGenerator(model, tokenizer, Device) : null;
        ModelSource = loaded is null ? null : source;
    }

    private void WarnIfInterrupted()
    {
        if (Checkpoint.ExistsIn(_checkpointFolder))
            Messages.Warning("Un entraînement interrompu existait (reprenable avec /resume). Il est remplacé par ce nouvel entraînement.");
    }

    // Chaque entraînement laisse deux fichiers avec la même date dans results/ :
    //   <date>.txt       les réponses aux questions de test
    //   <date>-loss.csv  les loss mesurées pendant l'entraînement
    private (TrainingLog Log, Checkpoint Checkpoint, string AnswersPath) NewRun(string runId, string answersName) =>
        (new TrainingLog(Path.Combine(_resultsFolder, $"{runId}-loss.csv"), _questionsPath),
         new Checkpoint(_checkpointFolder, runId),
         Path.Combine(_resultsFolder, $"{answersName}.txt"));

    /// <summary>
    /// Fin d'un entraînement complet : plus rien à reprendre, le même examen qu'après chaque entraînement
    /// (pour les comparer entre eux), puis le nouveau modèle devient celui du chat.
    /// </summary>
    private void Finish(MiniGpt model, BpeTokenizer tokenizer, string training, TrainingLog log, Checkpoint checkpoint,
        string answersPath, bool pretrainOnly)
    {
        if (Stopped(model, tokenizer, pretrainOnly))
            return;

        checkpoint.Delete();

        if (OnGpu && GpuMemory.Read() is var (usedMb, totalMb))
            Messages.Dim($"Mémoire GPU utilisée : {usedMb:N0} / {totalMb:N0} Mo (toute la carte).");

        string report = Messages.Spin("Questions de test...", () => QuestionTest.Run(
            new TextGenerator(model, tokenizer, Device), _questionsPath, answersPath,
            $"Entraînement : {training} | {model.ParameterCount:N0} paramètres | loss : {Path.GetFileName(log.LossCsvPath)}"));
        ShowReport(report, answersPath);

        // Après un --continue du pré-entraînement, le modèle en mémoire ne sait pas encore discuter.
        if (pretrainOnly)
            Messages.Info("Pour que le chat profite de ce pré-entraînement : /finetune (refait le fine-tuning depuis 'model-pretrain').");
        Adopt(model, tokenizer, pretrainOnly);
        Messages.Success("Entraînement terminé.");
    }

    private static void ShowReport(string report, string path)
    {
        AnsiConsole.WriteLine();
        Messages.Block("Questions de test", report);
        Messages.Dim($"(réponses écrites dans {path})");
    }

    // Nouveau modèle : entraîne le tokenizer BPE, puis crée le réseau (poids au hasard).
    private (MiniGpt, BpeTokenizer) CreateModel()
    {
        manual_seed(TrainingConfig.Pretraining.Seed);

        // Le tokenizer apprend sur les DEUX fichiers : il connaît les mots des dialogues,
        // et le modèle garde le même vocabulaire (donc la même taille de sortie) entre les deux phases.
        var tokenizer = Messages.Spin($"Entraînement du tokenizer BPE ({TokenizerVocabSize} tokens, sur {TokenizerSampleMb} Mo de texte.txt au plus)...",
            () => BpeTokenizer.Train(ReadSample(_pretrainPath, TokenizerSampleMb) + "\n" + ReadDialogues(), TokenizerVocabSize));
        var model = new MiniGpt(new ModelConfig { VocabSize = tokenizer.VocabSize }).to(Device);

        Messages.Info($"Vocabulaire : {tokenizer.VocabSize} tokens ({tokenizer.Alphabet.Length} caractères de base + {tokenizer.Merges.Count} fusions " +
                      $"+ {tokenizer.SpecialTokens.Count} tokens spéciaux : {string.Join(" ", tokenizer.SpecialTokens)}).");
        Messages.Info($"Modèle : {model.ParameterCount:N0} paramètres.");
        return (model, tokenizer);
    }

    // Phase 1 : apprend la langue au modèle sur texte.txt.
    // continuation non null (--continue) : store n'est remplacé que si la loss de vérification a baissé.
    private void Pretrain(MiniGpt model, BpeTokenizer tokenizer, ModelStore store, TrainingConfig config,
        TrainingLog log, Checkpoint checkpoint, CheckpointState? resume,
        ContinueState? continuation = null, TrainingInfo? previous = null)
    {
        // À la reprise, graine décalée du pas : on ne rejoue pas les mêmes extraits qu'au début.
        // Une continuation part aussi de son premier pas (sinon elle retirerait les extraits de --train).
        int seed = config.Seed + (resume?.Step ?? config.FirstStep);
        manual_seed(seed);

        // using : les tokens sont sur le GPU (~0,9 Go), libérés à la fin de la phase.
        using var dataset = GeneralText(tokenizer, seed);
        Messages.Info($"texte.txt : {dataset.Characters:N0} caractères -> {dataset.TrainLength + dataset.ValidationLength:N0} tokens.");
        var result = TrainPhase("Phase 1 - pré-entraînement", "pretrain", dataset, model, tokenizer, config, log,
            checkpoint, resume, optimizerPath: continuation is null ? null : store.OptimizerPath);

        SaveResult(store, PretrainedModel, model, tokenizer, result, checkpoint.RunId, continuation, previous);
    }

    // Phase 2 : spécialise le modèle (déjà pré-entraîné) sur les dialogues.
    // Le modèle n'est noté que sur les réponses "IA:", et 20 % de chaque batch vient de texte.txt
    // pour qu'il n'oublie pas le français général.
    private void FineTune(MiniGpt model, BpeTokenizer tokenizer, ModelStore store, TrainingConfig config,
        TrainingLog log, Checkpoint checkpoint, CheckpointState? resume,
        ContinueState? continuation = null, TrainingInfo? previous = null)
    {
        int seed = config.Seed + (resume?.Step ?? config.FirstStep);
        manual_seed(seed);

        using var dialogues = Dialogues(tokenizer, seed);
        // Autre graine : les extraits de texte général ne suivent pas le même tirage que les dialogues.
        using var generalText = GeneralText(tokenizer, seed + 1);

        var result = TrainPhase($"Phase 2 - fine-tuning (loss sur les réponses IA, {config.GeneralTextRatio:P0} de texte général)",
            "finetune", dialogues, model, tokenizer, config, log, checkpoint, resume,
            optimizerPath: continuation is null ? null : store.OptimizerPath, generalText);

        SaveResult(store, ChatModel, model, tokenizer, result, checkpoint.RunId, continuation, previous);
    }

    // --continue / --continue-finetune : steps pas de plus sur un modèle déjà entraîné, avec les données actuelles
    // de data/ et l'optimiseur sauvegardé avec lui. Les pas continuent la numérotation (et le CSV) d'avant.
    private void ContinuePhase(ContinueState continuation, MiniGpt model, BpeTokenizer tokenizer, ModelStore store,
        TrainingInfo? previous, TrainingLog log, Checkpoint checkpoint, CheckpointState? resume)
    {
        var config = BaseConfig(continuation.Phase).Continuation(continuation.FirstStep, continuation.Steps, continuation.LearningRate);
        Messages.Info($"Continuation : pas {config.FirstStep} -> {config.Steps}, warmup {config.WarmupSteps} pas, " +
                      $"vitesse {config.LearningRate:0.0e0} -> {config.MinLearningRate:0.0e0}.");

        if (continuation.Phase == "pretrain")
            Pretrain(model, tokenizer, store, config, log, checkpoint, resume, continuation, previous);
        else
            FineTune(model, tokenizer, store, config, log, checkpoint, resume, continuation, previous);
    }

    // Sauvegarde le modèle avec l'état d'AdamW et l'avancement (pas atteint, loss, date du CSV).
    // Une continuation ne remplace l'ancien modèle que si sa loss de vérification est meilleure. Les deux loss sont
    // mesurées sur les données actuelles (l'ancien modèle au premier pas) : celle notée à l'ancienne sauvegarde
    // a pu l'être sur d'autres données, elle n'est affichée que pour info.
    private static void SaveResult(ModelStore store, string name, MiniGpt model, BpeTokenizer tokenizer, TrainingResult result,
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

    // Les tokens de texte.txt et de dialogues.txt, relus depuis cache/ s'ils y sont pour ce fichier et ce tokenizer
    // (voir TokenCache) : texte.txt n'est découpé qu'une fois, pas à chaque phase ni à chaque lancement.
    private TextDataset GeneralText(BpeTokenizer tokenizer, int seed) =>
        Dataset("texte.txt", _pretrainPath, tokenizer, seed, onlyReplies: false,
            () => TextDataset.TokenizeFile(_pretrainPath, tokenizer));

    private TextDataset Dialogues(BpeTokenizer tokenizer, int seed) =>
        Dataset("dialogues.txt", _dialoguesPath, tokenizer, seed, onlyReplies: true,
            () => TextDataset.Tokenize(ReadDialogues(), tokenizer, onlyReplies: true));

    private TextDataset Dataset(string name, string path, BpeTokenizer tokenizer, int seed, bool onlyReplies,
        Func<TokenizedText> tokenize)
    {
        var (text, fromCache) = Messages.Spin($"Lecture de {name}...",
            () => TokenCache.Load(_cacheFolder, CheckExists(path), tokenizer, onlyReplies, tokenize));
        Messages.Dim(fromCache ? $"{name} : tokens relus depuis cache/." : $"{name} : découpé en tokens, gardé dans cache/.");
        return new TextDataset(text, seed, Device);
    }

    private TrainingResult TrainPhase(string name, string phase, TextDataset dataset, MiniGpt model, BpeTokenizer tokenizer,
        TrainingConfig config, TrainingLog log, Checkpoint checkpoint, CheckpointState? resume,
        string? optimizerPath = null, TextDataset? generalText = null)
    {
        // Lu dès le début : si le fichier de questions manque, on le sait tout de suite, pas après des heures.
        string sampleQuestion = QuestionTest.ReadQuestions(log.QuestionsPath)[0];
        var generator = new TextGenerator(model, tokenizer, Device);

        Messages.Title(name);
        Messages.Info($"{dataset.TrainLength + dataset.ValidationLength:N0} tokens, {config.Steps - config.FirstStep} étapes.");
        return TrainingProgressView.Run(config.FirstStep, config.Steps, progress =>
            new Trainer(model, dataset, config, Device, checkpoint, progress, generalText, phase, log.LossCsvPath,
                showSample: step => progress.Log(QuestionTest.Sample(generator, sampleQuestion, step)))
                .Train(resume, optimizerPath, _cancellation));
    }

    private ModelStore StoreFor(string phase) => phase == "pretrain" ? _pretrainStore : _store;

    private static string StoreName(string phase) => phase == "pretrain" ? PretrainedModel : ChatModel;

    private static TrainingConfig BaseConfig(string phase) =>
        phase == "pretrain" ? TrainingConfig.Pretraining : TrainingConfig.FineTuning;

    private static string Now() => DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");

    // Lit un fichier de données. Sur Windows les fins de ligne sont "\r\n" : on garde seulement "\n".
    private static string ReadText(string path) => File.ReadAllText(CheckExists(path)).Replace("\r\n", "\n");

    // dialogues.txt (Toi:/IA:) au format du modèle, avec ses tokens spéciaux (voir ChatFormat).
    private string ReadDialogues() => ChatFormat.FromDialogueFile(ReadText(_dialoguesPath));

    private static string CheckExists(string path) =>
        File.Exists(path) ? path : throw new UserException($"Fichier de données introuvable : {path}");

    // Un fichier de plus de maxMb Mo : maxMb blocs de 1 Mo pris à intervalles réguliers, coupés aux fins de ligne.
    // Réparti dans tout le fichier, l'échantillon voit chaque source (texte.txt en colle souvent plusieurs bout à bout).
    // Un fichier plus petit est lu en entier.
    private static string ReadSample(string path, int maxMb)
    {
        const int BlockSize = 1 << 20;
        long length = new FileInfo(CheckExists(path)).Length;
        if (length <= (long)maxMb * BlockSize)
            return ReadText(path);

        var sample = new StringBuilder();
        var buffer = new byte[BlockSize];
        using var file = File.OpenRead(path);
        for (int i = 0; i < maxMb; i++)
        {
            file.Position = length / maxMb * i;
            int read = file.ReadAtLeast(buffer, BlockSize, throwOnEndOfStream: false);
            // Le bloc commence et finit au milieu d'une ligne (voire d'un caractère UTF-8) : on garde les lignes entières.
            string block = Encoding.UTF8.GetString(buffer, 0, read);
            int first = block.IndexOf('\n') + 1;
            int last = block.LastIndexOf('\n') + 1;
            if (last > first)
                sample.Append(block, first, last - first);
        }
        return sample.Replace("\r\n", "\n").ToString();
    }
}

/// <summary>
/// Un modèle sauvegardé, pour /models. ParameterCount est déduit de la taille des poids (sans les charger).
/// Info = pas, loss de vérification et date de l'entraînement ; null pour un modèle d'avant --continue.
/// </summary>
public sealed record SavedModel(string Name, ModelConfig Config, long ParameterCount, long WeightsBytes,
    DateTime SavedAt, TrainingInfo? Info);

/// <summary>Où un entraînement écrit son suivi : le CSV des loss, et les questions pour les exemples.</summary>
public sealed record TrainingLog(string LossCsvPath, string QuestionsPath);
