using MiniChat.Data;
using MiniChat.Training;
using Xunit;
using static MiniChat.Tokenization.BpeTokenizer;
using static TorchSharp.torch;

namespace MiniChat.Tests;

/// <summary>La loss de vérification par caractère, et le fine-tuning sur des extraits de conversations.</summary>
public sealed class PerCharacterLossTests
{
    [Fact]
    public void TokensCountTheirTextCharactersAndSpecialTokensNone()
    {
        var tokenizer = TestData.Tokenizer;
        using var scope = NewDisposeScope();
        var characters = Trainer.TokenCharacters(tokenizer, CPU).data<float>().ToArray();

        Assert.Equal(tokenizer.VocabSize, characters.Length);
        foreach (string special in DefaultSpecialTokens)
            Assert.Equal(0, characters[tokenizer.SpecialId(special)]);
        // Un texte sans token spécial : autant de caractères que de lettres.
        Assert.Equal(TestData.Text.Length, tokenizer.Encode(TestData.Text).Sum(id => characters[id]));
    }

    [Fact]
    public void TrainingWritesThePerCharacterLossInTheCsv()
    {
        using var folder = new TempFolder();
        string csv = folder.File("loss.csv");
        using var dialogues = new TextDataset(TestData.Conversations(400), seed: 42, CPU,
            documentToken: TestData.Tokenizer.SpecialId(DocToken));
        using var model = TestData.TinyModel();
        var checkpoint = new Checkpoint(folder.File("checkpoint"), "run");
        checkpoint.Start(model, TestData.Tokenizer);

        var result = new Trainer(model, dialogues, TestData.TinyTraining(steps: 10), CPU, checkpoint, new RecordingProgress(),
            phase: "finetune", lossCsvPath: csv, tokenizer: TestData.Tokenizer).Train();

        var points = LossCsv.Read(csv).Single().Points;
        Assert.All(points, p => Assert.NotNull(p.ValLossPerChar));
        Assert.All(points, p => Assert.True(float.IsFinite(p.ValLoss) && float.IsFinite(p.ValLossPerChar!.Value)));
        // Plusieurs caractères par token : la loss par caractère est plus petite que par token.
        Assert.All(points, p => Assert.True(p.ValLossPerChar < p.ValLoss));
        Assert.Equal(points.MinBy(p => p.ValLoss)!.ValLossPerChar!.Value, result.BestValLossPerChar!.Value, precision: 3);   // CSV : 4 décimales
    }

    [Fact]
    public void OldCsvWithoutThePerCharacterColumnStillReads()
    {
        using var folder = new TempFolder();
        string csv = folder.File("loss.csv");
        File.WriteAllText(csv, "phase,etape,loss_entrainement,loss_verification,learning_rate,secondes,tokens_par_seconde,memoire_gpu_mo,passages\n" +
                               "pretrain,0,9.1,9.2,1E-05,0,,,0.000\n" +
                               "pretrain,500,4.1,4.2,0.001,60,40000,5000,0.020,1.2345\n");

        var points = LossCsv.Read(csv).Single().Points;
        Assert.Null(points[0].ValLossPerChar);
        Assert.Equal(1.2345f, points[1].ValLossPerChar);
    }
}
