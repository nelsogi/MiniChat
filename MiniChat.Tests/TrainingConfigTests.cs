using MiniChat.App;
using MiniChat.Config;
using MiniChat.Data;
using MiniChat.Training;
using Xunit;
using static TorchSharp.torch;

namespace MiniChat.Tests;

public sealed class LearningRateTests
{
    // Warmup de 10 pas jusqu'à 1e-3, puis cosinus jusqu'à 1e-4 au pas 110.
    private static readonly TrainingConfig Config = new()
    {
        Epochs = null, Steps = 110, LearningRate = 1e-3, MinLearningRate = 1e-4, WarmupSteps = 10,
    };

    [Theory]
    [InlineData(0, 1e-4)]   // 1/10 de la vitesse max dès le premier pas (jamais 0)
    [InlineData(4, 5e-4)]
    [InlineData(9, 1e-3)]   // fin du warmup : vitesse max
    public void WarmupRisesLinearly(int step, double expected) =>
        Assert.Equal(expected, Config.LearningRateAt(step), 12);

    [Fact]
    public void CosineGoesFromMaxToMin()
    {
        Assert.Equal(1e-3, Config.LearningRateAt(10), 12);                 // début du cosinus
        Assert.Equal((1e-3 + 1e-4) / 2, Config.LearningRateAt(60), 12);    // milieu
        Assert.Equal(1e-4, Config.LearningRateAt(110), 12);                // dernier pas prévu
    }

    [Fact]
    public void AfterTheLastStepItStaysAtMin() => Assert.Equal(1e-4, Config.LearningRateAt(500), 12);

    [Fact]
    public void CosineOnlyGoesDown()
    {
        for (int step = 11; step <= 110; step++)
            Assert.True(Config.LearningRateAt(step) < Config.LearningRateAt(step - 1));
    }

    [Fact]
    public void ContinuationHasItsOwnWarmupAndCosine()
    {
        var continuation = TrainingConfig.Pretraining.Continuation(firstStep: 1000, steps: 400);

        Assert.Equal(1000, continuation.FirstStep);
        Assert.Equal(1400, continuation.Steps);
        Assert.Equal(TrainingConfig.ContinueWarmupSteps, continuation.WarmupSteps);
        Assert.Equal(TrainingConfig.ContinueLearningRate / TrainingConfig.ContinueWarmupSteps, continuation.LearningRateAt(1000), 12);
        Assert.Equal(TrainingConfig.ContinueLearningRate, continuation.LearningRateAt(1000 + TrainingConfig.ContinueWarmupSteps - 1), 12);
        Assert.Equal(TrainingConfig.ContinueLearningRate / 10, continuation.LearningRateAt(1400), 12);
    }

    [Fact]
    public void ShortContinuationWarmsUpForHalfItsSteps()
    {
        var continuation = TrainingConfig.FineTuning.Continuation(firstStep: 0, steps: 100, learningRate: 5e-5);
        Assert.Equal(50, continuation.WarmupSteps);
        Assert.Equal(5e-5, continuation.LearningRate);
        Assert.Equal(5e-6, continuation.MinLearningRate);
    }

    [Fact]
    public void ContinuationNeverGoesFasterThanThePhase() =>
        Assert.Equal(TrainingConfig.FineTuning.LearningRate, TrainingConfig.FineTuning.Continuation(0, 1000).LearningRate);

    [Fact]
    public void TrainerFollowsTheSchedule()
    {
        // Le Trainer annonce à chaque pas la vitesse qu'il utilise : celle de LearningRateAt.
        using var folder = new TempFolder();
        using var dataset = new TextDataset(TestData.RandomTokens(5_000), seed: 42, CPU);
        using var model = TestData.TinyModel();
        var config = TestData.TinyTraining(steps: 12);
        var progress = new RecordingProgress();

        new Trainer(model, dataset, config, CPU, new Checkpoint(folder.Path, "test"), progress).Train();

        Assert.Equal(Enumerable.Range(0, 13), progress.Steps.Select(s => s.Step));
        Assert.All(progress.Steps, s => Assert.Equal(config.LearningRateAt(s.Step), s.LearningRate));
    }
}

public sealed class EpochStepsTests
{
    // small : 16 extraits x 4 petits batchs x 256 tokens = 16 384 tokens par étape.
    private const int BlockSize = 256;
    private const long TokensPerStep = 16 * 4 * BlockSize;

    [Fact]
    public void PretrainingReadsSixteenThousandTokensPerStep() =>
        Assert.Equal(TokensPerStep, TrainingConfig.Pretraining.MainTokensPerStep(BlockSize));

    [Theory]
    [InlineData(1.0, 100, 100)]
    [InlineData(2.0, 100, 200)]
    [InlineData(0.5, 101, 51)]    // 50,5 pas : arrondi au pas du dessus
    [InlineData(0.5, 1, 1)]       // jamais moins d'un pas
    public void StepsComeFromEpochsAndDataSize(double epochs, int stepsOfData, int expected)
    {
        var config = TrainingConfig.Pretraining.WithEpochs(epochs).ForData(stepsOfData * TokensPerStep, BlockSize);
        Assert.Equal(expected, config.Steps);
    }

    [Fact]
    public void DefaultEpochsAreKeptWithoutOption()
    {
        Assert.Equal(TrainingConfig.Pretraining.Epochs, TrainingConfig.Pretraining.WithEpochs(null).Epochs);
        Assert.Equal(TrainingConfig.FineTuning.Epochs, TrainingConfig.FineTuning.WithEpochs(null).Epochs);
        Assert.Equal(0.3, TrainingConfig.FineTuning.WithEpochs(0.3).Epochs);
    }

    [Fact]
    public void FineTuningCountsOnlyDialogueTokens()
    {
        // 20 % de texte général : 3 extraits sur 16, donc 13 x 4 x 256 tokens de dialogues par étape.
        var fineTune = TrainingConfig.FineTuning;
        Assert.Equal(3, fineTune.GeneralExtracts);
        Assert.Equal(13 * 4 * BlockSize, fineTune.MainTokensPerStep(BlockSize));
        Assert.Equal(150, fineTune.ForData(100 * 13 * 4 * BlockSize, BlockSize).Steps);   // 1,5 passage
    }

    [Fact]
    public void StepsCountFromTheFirstStep()
    {
        var config = TrainingConfig.Pretraining with { FirstStep = 1000, Epochs = 1.0 };
        Assert.Equal(1100, config.ForData(100 * TokensPerStep, BlockSize).Steps);
    }

    [Fact]
    public void WithoutEpochsStepsAreKept()
    {
        var continuation = TrainingConfig.Pretraining.Continuation(firstStep: 500, steps: 300);
        Assert.Equal(800, continuation.ForData(100 * TokensPerStep, BlockSize).Steps);
    }

    [Fact]
    public void EstimateUsesTheSameCount() =>
        Assert.Equal(200, TrainingEstimate.Steps(TrainingConfig.Pretraining.WithEpochs(2.0), 100 * TokensPerStep, BlockSize));

    [Fact]
    public void PlannedStepsComeFromTheTrainingPartOfTheData()
    {
        using var dataset = new TextDataset(TestData.RandomTokens(20_000), seed: 42, CPU);
        using var model = TestData.TinyModel();
        var config = TestData.TinyTraining(steps: 0) with { Epochs = 2.0 };

        var planned = TrainingPhase.Planned(config, dataset, model, resume: null);

        long perStep = config.MainTokensPerStep(model.Config.BlockSize);
        Assert.Equal((int)Math.Ceiling(2.0 * dataset.TrainLength / perStep), planned.Steps);
    }

    [Fact]
    public void ResumeKeepsThePlannedLastStep()
    {
        // Une reprise finit là où c'était prévu (même cosinus), même si les données ont changé depuis.
        using var dataset = new TextDataset(TestData.RandomTokens(20_000), seed: 42, CPU);
        using var model = TestData.TinyModel();
        var config = TestData.TinyTraining(steps: 0) with { Epochs = 2.0 };

        var planned = TrainingPhase.Planned(config, dataset, model, new CheckpointState("pretrain", 300, "run", Steps: 777));

        Assert.Equal(777, planned.Steps);
    }
}
