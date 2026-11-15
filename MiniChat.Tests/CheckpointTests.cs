using MiniChat.Data;
using MiniChat.Model;
using MiniChat.Training;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace MiniChat.Tests;

public sealed class CheckpointTests
{
    [Fact]
    public void ResumeGetsTheSameWeightsOptimizerAndStep()
    {
        using var folder = new TempFolder();
        using var dataset = new TextDataset(TestData.RandomTokens(5_000), seed: 42, CPU);
        using var model = TestData.TinyModel();
        var optimizer = optim.AdamW(model.parameters(), lr: 1e-3);
        for (int i = 0; i < 3; i++)
            Step(model, optimizer, dataset);

        var checkpoint = new Checkpoint(folder.Path, "2026-01-01_00-00-00");
        checkpoint.Start(model, TestData.Tokenizer);
        var state = new CheckpointState("finetune", 3, checkpoint.RunId, StartValLoss: 4.5f, Steps: 50);
        checkpoint.SaveProgress(model, optimizer, state);

        // Même étape.
        Assert.True(Checkpoint.ExistsIn(folder.Path));
        Assert.Equal(state, Checkpoint.ReadState(folder.Path));

        // Mêmes poids.
        var (loaded, tokenizer) = new Checkpoint(folder.Path, state.RunId).LoadModel(CPU);
        using (loaded)
        {
            AssertSameWeights(model, loaded);
            Assert.Equal(TestData.Tokenizer.Merges, tokenizer.Merges);
            Assert.Equal(model.Config, loaded.Config);

            // Même état d'AdamW : resauvegardé, il donne le même fichier...
            var loadedOptimizer = optim.AdamW(loaded.parameters(), lr: 1e-3);
            checkpoint.LoadOptimizer(loadedOptimizer);
            optimizer.save_state_dict(folder.File("a.dat"));
            loadedOptimizer.save_state_dict(folder.File("b.dat"));
            Assert.Equal(File.ReadAllBytes(folder.File("a.dat")), File.ReadAllBytes(folder.File("b.dat")));

            // ... et le pas suivant, sur le même batch, mène aux mêmes poids (moyennes d'AdamW comprises).
            // Sans dropout : ses masques sont tirés au hasard, ils seraient différents pour les deux modèles.
            var (x, y, _) = dataset.FixedValidation(4, model.Config.BlockSize);
            model.DropoutRate.Value = loaded.DropoutRate.Value = 0;
            Step(model, optimizer, x, y);
            Step(loaded, loadedOptimizer, x, y);
            AssertSameWeights(model, loaded);
        }
    }

    [Fact]
    public void BestWeightsAndScoreAreKeptTogether()
    {
        using var folder = new TempFolder();
        using var model = TestData.TinyModel(seed: 1);
        using var other = TestData.TinyModel(seed: 2);
        var checkpoint = new Checkpoint(folder.Path, "run");
        checkpoint.Start(model, TestData.Tokenizer);

        checkpoint.SaveBest(model, new BestScore(3.25f, 1500));
        checkpoint.LoadBest(other);

        AssertSameWeights(model, other);
        Assert.Equal(new BestScore(3.25f, 1500), checkpoint.ReadBest());
    }

    [Fact]
    public void ContinuationSettingsAreKeptForResume()
    {
        using var folder = new TempFolder();
        using var model = TestData.TinyModel();
        var continuation = new ContinueState("pretrain", 9137, 2000, 3e-4);

        new Checkpoint(folder.Path, "run").Start(model, TestData.Tokenizer, continuation);

        Assert.Equal(continuation, new Checkpoint(folder.Path, "run").ReadContinuation());
    }

    [Fact]
    public void StoppedTrainingResumesAtTheStepItReached()
    {
        using var folder = new TempFolder();
        using var dataset = new TextDataset(TestData.RandomTokens(5_000), seed: 42, CPU);
        using var model = TestData.TinyModel();
        var config = TestData.TinyTraining(steps: 12);
        var checkpoint = new Checkpoint(folder.Path, "run");
        checkpoint.Start(model, TestData.Tokenizer);

        // Échap au pas 7 : l'avancement est sauvegardé à ce pas exact.
        using var stop = new CancellationTokenSource();
        var progress = new RecordingProgress(step => { if (step == 7) stop.Cancel(); });
        new Trainer(model, dataset, config, CPU, checkpoint, progress, phase: "pretrain").Train(cancellation: stop.Token);

        var state = Checkpoint.ReadState(folder.Path);
        Assert.Equal(("pretrain", 7, "run", 12), (state.Phase, state.Step, state.RunId, state.Steps));
        Assert.Equal(progress.Reports[0].ValLoss, state.StartValLoss);

        // La reprise repart du pas 7 (aucun pas refait), avec les poids du checkpoint, et va jusqu'au pas prévu.
        var (resumed, _) = checkpoint.LoadModel(CPU);
        using (resumed)
        {
            var resumedProgress = new RecordingProgress();
            var result = new Trainer(resumed, dataset, config, CPU, checkpoint, resumedProgress, phase: "pretrain")
                .Train(resume: state);

            Assert.Equal(Enumerable.Range(7, 6), resumedProgress.Steps.Select(s => s.Step));
            Assert.Equal(12, result.LastStep);
            Assert.Equal(state.StartValLoss, result.StartValLoss);
            Assert.Contains(resumedProgress.Messages, m => m.StartsWith("Reprise au pas 7"));
        }
    }

    private static void Step(MiniGpt model, optim.Optimizer optimizer, TextDataset dataset)
    {
        using var scope = NewDisposeScope();
        var (x, y, _) = dataset.GetBatch(validation: false, 4, model.Config.BlockSize, CPU);
        Step(model, optimizer, x, y);
    }

    private static void Step(MiniGpt model, optim.Optimizer optimizer, Tensor x, Tensor y)
    {
        using var scope = NewDisposeScope();
        optimizer.zero_grad();
        var logits = model.forward(x);
        nn.functional.cross_entropy(logits.view(-1, model.Config.VocabSize), y.view(-1)).backward();
        optimizer.step();
    }

    private static void AssertSameWeights(MiniGpt expected, MiniGpt actual)
    {
        var a = expected.state_dict();
        var b = actual.state_dict();
        Assert.Equal(a.Keys, b.Keys);
        Assert.All(a.Keys, name => Assert.True(a[name].equal(b[name]).all().item<bool>(), $"{name} diffère"));
    }
}
