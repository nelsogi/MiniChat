using System.Text;
using MiniChat.Config;
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
/// Le détail est ailleurs : les fichiers de données (DatasetFactory), les phases (PretrainPhase, FinetunePhase,
/// ContinuePhase), les modèles sauvegardés (ModelCatalog) et /export (ExportService).
///
/// Deux dossiers :
///   data/    texte.txt, dialogues.txt, questions-test.txt
///   models/  model/, model-pretrain/, et ce que l'entraînement écrit : checkpoint/, results/
/// </summary>
public sealed class Workbench
{
    public const string ChatModel = ModelCatalog.ChatModel;
    public const string PretrainedModel = ModelCatalog.PretrainedModel;

    private readonly string _checkpointFolder;   // sauvegarde régulière de l'entraînement en cours (pour /resume)
    private readonly string _resultsFolder;

    private readonly DatasetFactory _data;
    private readonly ModelCatalog _models;
    private readonly ExportService _export;
    private readonly PretrainPhase _pretrain;
    private readonly FinetunePhase _finetune;
    private readonly ContinuePhase _continue;
    private readonly EvalHistory _evalHistory;

    // L'arrêt demandé pour l'entraînement en cours (voir Training), vide le reste du temps.
    private CancellationToken _cancellation;

    // Le tokenizer de Model (pour /save).
    private BpeTokenizer? _tokenizer;

    public Workbench(string dataFolder, string modelsFolder, Device device)
    {
        DataFolder = dataFolder;
        ModelsFolder = modelsFolder;
        Device = device;
        _checkpointFolder = Path.Combine(modelsFolder, "checkpoint");
        _resultsFolder = Path.Combine(modelsFolder, "results");
        _data = new DatasetFactory(dataFolder, Path.Combine(modelsFolder, "cache"), device);
        _models = new ModelCatalog(modelsFolder);
        _export = new ExportService(_models, Path.Combine(modelsFolder, "exports"));
        _evalHistory = new EvalHistory(_resultsFolder);

        var phase = new TrainingPhase(device);
        _pretrain = new PretrainPhase(_data, phase);
        _finetune = new FinetunePhase(_data, phase);
        _continue = new ContinuePhase(_pretrain, _finetune);
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
        var (store, name) = DefaultModel == PretrainedModel && _models.Pretrained.Exists
            ? (_models.Pretrained, PretrainedModel)
            : (_models.Chat, ChatModel);
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

    /// <summary>
    /// Ce que /train --size va faire, avant de le lancer : paramètres, pas et durée estimée de chaque phase.
    /// Nombre de tokens estimé d'après la taille des fichiers (le tokenizer n'est pas encore entraîné).
    /// </summary>
    public TrainingPlan Plan(string size, double? epochs)
    {
        var config = SizeConfig(size);
        long parameters;
        using (var model = new MiniGpt(config))   // sur le CPU, juste pour compter
            parameters = model.ParameterCount;

        var pretrain = TrainingConfig.PretrainingFor(config).WithEpochs(epochs);
        var fineTune = TrainingConfig.FineTuningFor(config);
        int pretrainSteps = TrainingEstimate.Steps(pretrain, TrainingEstimate.TokensIn(DatasetFactory.CheckExists(_data.TextPath)), config.BlockSize);
        int fineTuneSteps = TrainingEstimate.Steps(fineTune, TrainingEstimate.TokensIn(DatasetFactory.CheckExists(_data.DialoguesPath)), config.BlockSize);
        return new TrainingPlan(size, config, parameters, pretrain, pretrainSteps,
            TrainingEstimate.Duration(config, parameters, pretrain, pretrainSteps), fineTuneSteps,
            TrainingEstimate.Duration(config, parameters, fineTune, fineTuneSteps));
    }

    /// <summary>
    /// Nouveau modèle de la taille size (ModelConfig.Sizes) : pré-entraînement puis fine-tuning.
    /// epochs : passages sur texte.txt (--epochs), null = TrainingConfig.Pretraining.Epochs (1,1). Le nombre de pas
    /// en est calculé une fois texte.txt découpé (voir TrainingConfig.ForData).
    /// </summary>
    public void Train(string size = ModelConfig.DefaultSize, double? epochs = null, CancellationToken cancellation = default) =>
        Training(cancellation, () =>
    {
        WarnIfInterrupted();
        string now = Now();
        var run = NewRun(now, now);

        var (model, tokenizer) = CreateModel(SizeConfig(size));
        run.Checkpoint.Start(model, tokenizer);
        _pretrain.Run(model, tokenizer, _models.Pretrained, TrainingConfig.PretrainingFor(model.Config).WithEpochs(epochs), run, resume: null);
        if (Stopped(model, tokenizer, pretrainOnly: true)) return;
        _finetune.Run(model, tokenizer, _models.Chat, TrainingConfig.FineTuningFor(model.Config), run, resume: null);
        Finish(model, tokenizer, "pré-entraînement + fine-tuning", run, pretrainOnly: false);
    });

    /// <summary>
    /// Refait seulement la phase 2, en repartant du modèle pré-entraîné (pas du modèle final).
    /// epochs : passages sur les dialogues (--epochs), null = TrainingConfig.FineTuning.Epochs (1,5).
    /// </summary>
    public void FineTuneOnly(double? epochs = null, CancellationToken cancellation = default) => Training(cancellation, () =>
    {
        if (!_models.Pretrained.Exists)
            throw new UserException($"Pas de modèle pré-entraîné dans '{PretrainedModel}' : lance d'abord /train.");

        WarnIfInterrupted();
        string now = Now();
        var run = NewRun(now, now);

        var (model, tokenizer) = _models.Pretrained.Load(Device);
        Messages.Info($"Modèle pré-entraîné chargé depuis '{PretrainedModel}'.");
        run.Checkpoint.Start(model, tokenizer);
        _finetune.Run(model, tokenizer, _models.Chat, TrainingConfig.FineTuningFor(model.Config).WithEpochs(epochs), run, resume: null);
        Finish(model, tokenizer, "fine-tuning seul (--finetune)", run, pretrainOnly: false);
    });

    /// <summary>
    /// steps pas de plus sur un modèle déjà entraîné, avec les données actuelles de data/ (on peut en avoir ajouté).
    /// phase "pretrain" : depuis model-pretrain/ ; "finetune" : depuis model/.
    /// learningRate null : vitesse par défaut (voir TrainingConfig.Continuation).
    /// </summary>
    public void Continue(string phase, int steps, double? learningRate, CancellationToken cancellation = default) =>
        Training(cancellation, () =>
    {
        var store = _models.StoreFor(phase);
        if (!store.Exists)
            throw new UserException($"Rien à continuer : pas de modèle dans '{ModelCatalog.StoreName(phase)}'. Lance d'abord /train.");

        WarnIfInterrupted();
        // Le CSV des loss continue celui de l'entraînement d'origine (date gardée dans training.json),
        // mais les réponses aux questions ont leur propre fichier.
        var info = store.ReadInfo();
        string now = Now();
        var run = NewRun(info?.RunId ?? now, now);

        // Repart du modèle sauvegardé, avec SON tokenizer : jamais de nouveau BPE, sinon les ids ne voudraient
        // plus rien dire pour le modèle. Les caractères des nouvelles données qu'il ne connaît pas sont ignorés.
        var (model, tokenizer) = store.Load(Device);
        int firstStep = info?.Step ?? 0;
        var continuation = new ContinueState(phase, firstStep, steps,
            ContinuePhase.BaseConfig(phase, model.Config).Continuation(firstStep, steps, learningRate).LearningRate);

        Messages.Info($"Modèle chargé depuis '{ModelCatalog.StoreName(phase)}'" +
                      (info is null ? " (sauvegardé avant --continue : pas numérotés depuis 0)." : $", arrêté au pas {firstStep}."));
        run.Checkpoint.Start(model, tokenizer, continuation);
        _continue.Run(continuation, model, tokenizer, store, info, run, resume: null);
        Finish(model, tokenizer, phase == "pretrain"
                ? $"pré-entraînement continué de {steps} pas (--continue)"
                : $"fine-tuning continué de {steps} pas (--continue-finetune)",
            run, pretrainOnly: phase == "pretrain");
    });

    /// <summary>Reprend la phase en cours là où elle s'est arrêtée (poids, optimiseur, pas, meilleure loss).</summary>
    public void Resume(CancellationToken cancellation = default) => Training(cancellation, () =>
    {
        if (!Checkpoint.ExistsIn(_checkpointFolder))
            throw new UserException("Rien à reprendre : pas d'entraînement interrompu dans 'checkpoint'.");

        // Une reprise garde la date de l'entraînement d'origine : le CSV continue au même endroit.
        var state = Checkpoint.ReadState(_checkpointFolder);
        var continuation = new Checkpoint(_checkpointFolder, state.RunId).ReadContinuation();
        var run = NewRun(state.RunId, continuation is null ? state.RunId : Now());

        var (model, tokenizer) = run.Checkpoint.LoadModel(Device);
        Messages.Info($"Reprise de l'entraînement du {state.RunId} : phase '{state.Phase}', pas {state.Step}.");

        if (continuation is not null)
        {
            // Un --continue interrompu : mêmes réglages, et toujours rien qu'une phase.
            var store = _models.StoreFor(continuation.Phase);
            _continue.Run(continuation, model, tokenizer, store, store.ReadInfo(), run, state);
        }
        else if (state.Phase == "pretrain")
        {
            _pretrain.Run(model, tokenizer, _models.Pretrained, TrainingConfig.PretrainingFor(model.Config), run, state);
            if (Stopped(model, tokenizer, pretrainOnly: true)) return;
            _finetune.Run(model, tokenizer, _models.Chat, TrainingConfig.FineTuningFor(model.Config), run, resume: null);
        }
        else
        {
            _finetune.Run(model, tokenizer, _models.Chat, TrainingConfig.FineTuningFor(model.Config), run, state);
        }
        Finish(model, tokenizer, "entraînement repris (--resume)", run, pretrainOnly: continuation?.Phase == "pretrain");
    });

    /// <summary>Les questions de test sur le modèle chargé, sans rien entraîner. Réponses dans results/&lt;date&gt;-test.txt.</summary>
    public void RunTest(CancellationToken cancellation = default)
    {
        if (Model is not { } model || Generator is not { } generator)
            throw new UserException("Aucun modèle chargé : lance /train d'abord.");

        string path = Path.Combine(_resultsFolder, $"{Now()}-test.txt");
        string report = Messages.Spin("Questions de test... (Échap pour arrêter)", () => QuestionTest.Run(generator,
            _data.QuestionsPath, path, $"Test (/test) | modèle '{ModelSource}' | {model.ParameterCount:N0} paramètres", cancellation));
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
    public IReadOnlyList<SavedModel> ListModels() => _models.List();

    /// <summary>
    /// Garde le modèle chargé sous un nom (saved/&lt;nom&gt;/), pour le recharger ou le comparer plus tard,
    /// même après un nouvel entraînement. Un modèle du même nom est remplacé ; renvoie true dans ce cas.
    /// </summary>
    public bool SaveAs(string name)
    {
        if (Model is not { } model || _tokenizer is not { } tokenizer)
            throw new UserException("Aucun modèle chargé : rien à sauvegarder.");
        var saved = _models.SavedStore(name);

        // L'avancement (pas, loss, date de l'entraînement) suit le modèle ; l'état d'AdamW (deux fois la taille
        // du modèle) non : un modèle gardé sert à discuter et à comparer, pas à être continué.
        var info = ModelSource is { } source && _models.TryStore(source) is { Exists: true } from ? from.ReadInfo() : null;
        bool replaced = Directory.Exists(saved.Folder);
        if (replaced)
            Directory.Delete(saved.Folder, recursive: true);
        saved.Save(model, tokenizer, info: info);
        return replaced;
    }

    /// <summary>Charge un modèle sauvegardé (model, model-pretrain, ou un nom de /save) : il devient celui du chat.</summary>
    public void Load(string name)
    {
        var store = _models.Require(name);
        SetModel(null, null);   // libère la mémoire (GPU) de l'ancien avant de charger le nouveau
        try
        {
            SetModel(store.Load(Device), _models.NameOf(name));
        }
        finally
        {
            if (Model is null)
                LoadSaved();
        }
    }

    /// <summary>
    /// Les questions de test posées à deux modèles sauvegardés, dans les conditions de /test (température
    /// et graine fixes) : seule la différence entre les modèles compte. Puis l'évaluation notée (/eval) de chacun,
    /// gardée dans leur historique. cancellation : s'arrête avant la question suivante (sans évaluation).
    /// </summary>
    public CompareResult Compare(string first, string second, CancellationToken cancellation = default)
    {
        var (firstStore, secondStore) = (_models.Require(first), _models.Require(second));
        (first, second) = (_models.NameOf(first), _models.NameOf(second));
        var questions = QuestionTest.ReadQuestions(_data.QuestionsPath);
        var evalQuestions = EvalTest.ReadQuestions(_data.EvalPath);
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

                if (cancellation.IsCancellationRequested)
                    return new CompareResult(answers, null, null);
                var evalA = Evaluate(first, firstStore, a, evalQuestions, cancellation);
                var evalB = Evaluate(second, secondStore, b, evalQuestions, cancellation);
                return new CompareResult(answers, evalA, evalB);
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
    }

    /// <summary>
    /// L'évaluation notée (data/eval.json, voir EvalTest) du modèle chargé, ou d'un modèle sauvegardé (name).
    /// Détail dans results/&lt;date&gt;-eval-&lt;nom&gt;.txt ; le score s'ajoute à l'historique du modèle (si elle est complète).
    /// cancellation : s'arrête avant la question suivante.
    /// </summary>
    public EvalRun Eval(string? name, CancellationToken cancellation = default)
    {
        var questions = EvalTest.ReadQuestions(_data.EvalPath);
        if (name is null || (ModelSource is { } loaded && name.Equals(loaded, StringComparison.OrdinalIgnoreCase)))
        {
            if (Generator is not { } generator || ModelSource is not { } source)
                throw new UserException("Aucun modèle chargé : lance /train, ou /eval <nom> (/models pour la liste).");
            return Evaluate(source, _models.TryStore(source), generator, questions, cancellation);
        }

        var store = _models.Require(name);
        name = _models.NameOf(name);
        var (model, tokenizer) = Messages.Spin($"Chargement de '{name}'...", () => store.Load(Device));
        using (model)
            return Evaluate(name, store, new TextGenerator(model, tokenizer, Device), questions, cancellation);
    }

    /// <summary>L'historique des évaluations d'un modèle (voir EvalHistory), de la plus ancienne à la plus récente.</summary>
    public IReadOnlyList<EvalRecord> EvalHistoryOf(string name) => _evalHistory.Read(_models.NameOf(name));

    /// <summary>
    /// Exporte un modèle sauvegardé (model, model-pretrain, ou un nom de /save) en exports/&lt;nom&gt;.nelmodel,
    /// pour /import dans Nel Code (voir NELMODEL-FORMAT.md), puis relit le fichier pour le vérifier.
    /// Un export du même nom est remplacé.
    /// </summary>
    public ExportResult Export(string name, string source, string? description) => _export.Export(name, source, description);

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
        if (pretrainOnly && _models.Chat.Exists)
        {
            model.Dispose();
            SetModel(_models.Chat.Load(Device), ChatModel);
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
    private TrainingRun NewRun(string runId, string answersName) =>
        new(new TrainingLog(Path.Combine(_resultsFolder, $"{runId}-loss.csv"), _data.QuestionsPath),
            new Checkpoint(_checkpointFolder, runId),
            Path.Combine(_resultsFolder, $"{answersName}.txt"),
            _cancellation);

    /// <summary>
    /// Fin d'un entraînement complet : plus rien à reprendre, le même examen qu'après chaque entraînement
    /// (pour les comparer entre eux), puis le nouveau modèle devient celui du chat.
    /// </summary>
    private void Finish(MiniGpt model, BpeTokenizer tokenizer, string training, TrainingRun run, bool pretrainOnly)
    {
        if (Stopped(model, tokenizer, pretrainOnly))
            return;

        run.Checkpoint.Delete();

        if (OnGpu && GpuMemory.Read() is var (usedMb, totalMb))
            Messages.Dim($"Mémoire GPU utilisée : {usedMb:N0} / {totalMb:N0} Mo (toute la carte).");

        string report = Messages.Spin("Questions de test...", () => QuestionTest.Run(
            new TextGenerator(model, tokenizer, Device), _data.QuestionsPath, run.AnswersPath,
            $"Entraînement : {training} | {model.ParameterCount:N0} paramètres | loss : {Path.GetFileName(run.Log.LossCsvPath)}"));
        ShowReport(report, run.AnswersPath);

        // Après un --continue du pré-entraînement, le modèle en mémoire ne sait pas encore discuter.
        if (pretrainOnly)
            Messages.Info("Pour que le chat profite de ce pré-entraînement : /finetune (refait le fine-tuning depuis 'model-pretrain').");
        Adopt(model, tokenizer, pretrainOnly);
        Messages.Success("Entraînement terminé.");
    }

    // Pose les questions de eval.json à generator, écrit le détail dans results/, et garde le score dans l'historique
    // du modèle name (avec l'entraînement qui a produit ses poids : store.ReadInfo()).
    private EvalRun Evaluate(string name, ModelStore? store, TextGenerator generator, IReadOnlyList<EvalQuestion> questions,
        CancellationToken cancellation)
    {
        var history = _evalHistory.Read(name);
        var result = Messages.Spin($"Évaluation de '{name}'...", setStatus => EvalTest.Run(generator, questions,
            i => setStatus($"Évaluation de '{name}' : question {i + 1}/{questions.Count}... (Échap pour arrêter)"), cancellation));

        var info = store is { Exists: true } ? store.ReadInfo() : null;
        string path = Path.Combine(_resultsFolder, $"{Now()}-eval-{name}.txt");
        Directory.CreateDirectory(_resultsFolder);
        File.WriteAllText(path, EvalTest.Report(result, $"Modèle '{name}'" + (info is null ? "" :
            $" | entraînement du {info.RunId}, pas {info.Step}, loss de vérification {info.ValLoss:F3}")), new UTF8Encoding(false));

        if (result.Complete)
            _evalHistory.Add(new EvalRecord(DateTime.Now, name, info?.RunId, info?.Step, info?.ValLoss, result.Correct,
                result.Total, result.Scores));
        return new EvalRun(name, result, history, path);
    }

    private static void ShowReport(string report, string path)
    {
        AnsiConsole.WriteLine();
        Messages.Block("Questions de test", report);
        Messages.Dim($"(réponses écrites dans {path})");
    }

    // Une taille de ModelConfig.Sizes, avec le vocabulaire du tokenizer qui sera entraîné.
    private static ModelConfig SizeConfig(string size) =>
        ModelConfig.Sizes.TryGetValue(size, out var config)
            ? config with { VocabSize = DatasetFactory.TokenizerVocabSize }
            : throw new UserException($"Taille inconnue : « {size} ». Tailles : {string.Join(", ", ModelConfig.Sizes.Keys)}.");

    // Nouveau modèle : entraîne le tokenizer BPE, puis crée le réseau (poids au hasard).
    // Un nouveau tokenizer a une autre empreinte : les tokens de cache/ sont redécoupés (voir TokenCache).
    private (MiniGpt, BpeTokenizer) CreateModel(ModelConfig config)
    {
        manual_seed(TrainingConfig.Pretraining.Seed);

        var tokenizer = _data.TrainTokenizer();
        var model = new MiniGpt(config with { VocabSize = tokenizer.VocabSize }).to(Device);

        Messages.Info($"Vocabulaire : {tokenizer.VocabSize} tokens ({tokenizer.Alphabet.Length} caractères de base + {tokenizer.Merges.Count} fusions " +
                      $"+ {tokenizer.SpecialTokens.Count} tokens spéciaux : {string.Join(" ", tokenizer.SpecialTokens)}).");
        Messages.Info($"Modèle : {model.ParameterCount:N0} paramètres.");
        return (model, tokenizer);
    }

    private static string Now() => DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
}

/// <summary>
/// Ce que /train va faire (voir Workbench.Plan) : taille et réglages du modèle, paramètres, et pour chaque phase
/// le nombre de pas et la durée estimés (voir TrainingEstimate).
/// </summary>
public sealed record TrainingPlan(string Size, ModelConfig Config, long Parameters, TrainingConfig Pretraining,
    int PretrainSteps, TimeSpan PretrainDuration, int FineTuneSteps, TimeSpan FineTuneDuration);

/// <summary>
/// Une évaluation (/eval) d'un modèle : son résultat, les évaluations d'avant dans son historique (sans celle-ci),
/// et le fichier du détail (chaque question, la réponse, juste ou non).
/// </summary>
public sealed record EvalRun(string Model, EvalResult Result, IReadOnlyList<EvalRecord> History, string ReportPath)
{
    /// <summary>L'évaluation d'avant, null si c'est la première.</summary>
    public EvalRecord? Previous => History.Count > 0 ? History[^1] : null;
}

/// <summary>
/// /compare : les questions de test (question, réponse du premier, réponse du second), puis l'évaluation de chacun
/// (null si /compare a été arrêté avant).
/// </summary>
public sealed record CompareResult(IReadOnlyList<(string Question, string First, string Second)> Answers,
    EvalRun? FirstEval, EvalRun? SecondEval);
