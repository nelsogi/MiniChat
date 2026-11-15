using MiniChat.Data;
using Xunit;
using static MiniChat.Tokenization.BpeTokenizer;
using static TorchSharp.torch;

namespace MiniChat.Tests;

/// <summary>
/// Les extraits de dialogues (fine-tuning) : chacun commence au début d'une conversation, suivie d'autres conversations
/// entières, puis du remplissage ; chaque conversation garde son numéro, rien ne passe d'un côté à l'autre.
/// </summary>
public sealed class ConversationExtractTests
{
    private const int BlockSize = 32;

    private static readonly TokenizedText Text = TestData.Conversations(400);
    private static int Doc => TestData.Tokenizer.SpecialId(DocToken);

    private static TextDataset Dataset(int seed = 42) => new(Text, seed, CPU, documentToken: Doc);

    // Le remplissage : x = y = <|doc|>. Dans une conversation, <|doc|> est toujours suivi de <|user|>.
    private static Tensor Padding(Batch batch) => batch.X.eq(Doc).logical_and(batch.Y.eq(Doc));

    [Fact]
    public void ConversationsStartAtDocAndStayOnTheirSide()
    {
        var (train, validation) = TextDataset.Split(Text.Ids.Length, validationEvery: 10);
        foreach (var segments in new[] { train, validation })
        {
            var conversations = TextDataset.Conversations(Text.Ids, segments, (short)Doc);
            Assert.NotEmpty(conversations);
            foreach (var conversation in conversations)
            {
                Assert.Equal(Doc, Text.Ids[conversation.Start]);
                // Aucun <|doc|> à l'intérieur : une conversation s'arrête au début de la suivante.
                Assert.DoesNotContain(Text.Ids.Skip(conversation.Start + 1).Take(conversation.Length - 1), id => id == Doc);
                // Elle reste dans un segment de son côté.
                Assert.Contains(segments, s => conversation.Start >= s.Start
                                               && conversation.Start + conversation.Length <= s.Start + s.Length);
            }
        }

        // Aucune conversation des deux côtés à la fois.
        var trainStarts = TextDataset.Conversations(Text.Ids, train, (short)Doc).Select(c => c.Start);
        var validationStarts = TextDataset.Conversations(Text.Ids, validation, (short)Doc).Select(c => c.Start);
        Assert.Empty(trainStarts.Intersect(validationStarts));
    }

    [Fact]
    public void EveryExtractStartsWithAConversationThenWholeOnes()
    {
        using var dataset = Dataset();
        using var scope = NewDisposeScope();
        var batch = dataset.GetBatch(validation: false, 64, BlockSize, CPU);
        var x = batch.X.data<long>().ToArray();
        var y = batch.Y.data<long>().ToArray();
        var documents = batch.Documents!.data<long>().ToArray();
        var mask = batch.Mask.data<float>().ToArray();

        for (int b = 0; b < 64; b++)
        {
            int row = b * BlockSize;
            Assert.Equal(Doc, x[row]);                         // commence au début d'une conversation
            Assert.Equal(0, documents[row]);
            for (int t = 1; t < BlockSize; t++)
            {
                long document = documents[row + t], before = documents[row + t - 1];
                Assert.True(document == before || document == before + 1, "numéros de conversation croissants");
                if (document != before)
                {
                    // Une nouvelle conversation (ou le remplissage) commence par <|doc|>, et celle d'avant était
                    // entière : la dernière réponse attendue (y) était son dernier token, pas le <|doc|> de la suivante.
                    Assert.Equal(Doc, x[row + t]);
                    Assert.NotEqual(Doc, y[row + t - 1]);
                }
                else if (x[row + t] == Doc)
                {
                    Assert.Equal(Doc, y[row + t]);             // un <|doc|> au milieu d'un numéro : seulement le remplissage
                }
            }
        }
        // <|doc|> n'est jamais noté (ni dans les conversations, ni dans le remplissage).
        Assert.DoesNotContain(Enumerable.Range(0, y.Length), i => y[i] == Doc && mask[i] > 0);
    }

    [Fact]
    public void FirstConversationIsCutOnlyIfTooLongAndTheOthersAreWhole()
    {
        var conversations = TextDataset.Conversations(Text.Ids, TextDataset.Split(Text.Ids.Length, 10).Train, (short)Doc);
        Assert.Contains(conversations, c => c.Length - 1 > BlockSize);   // il y a bien des conversations à couper
        for (int first = 0; first < conversations.Length; first++)
        {
            var runs = TextDataset.Runs(conversations, first, BlockSize);
            int used = runs.Sum(r => r.Length);
            Assert.InRange(used, 1, BlockSize);
            Assert.Equal(Math.Min(conversations[first].Length - 1, BlockSize), runs[0].Length);
            for (int i = 1; i < runs.Count; i++)
                Assert.Equal(conversations[first + i] with { Length = conversations[first + i].Length - 1 }, runs[i]);
            // La suivante ne tenait pas en entier (sinon elle aurait été ajoutée).
            if (first + runs.Count < conversations.Length)
                Assert.True(conversations[first + runs.Count].Length - 1 > BlockSize - used);
        }
    }

    [Fact]
    public void PaddingIsNeverScoredAndIsItsOwnConversation()
    {
        using var dataset = Dataset();
        using var scope = NewDisposeScope();
        var batch = dataset.GetBatch(validation: false, 64, BlockSize, CPU);
        var padding = Padding(batch);

        Assert.True(padding.any().item<bool>());
        Assert.True(batch.Mask.masked_select(padding).eq(0).all().item<bool>());
        // Son numéro est celui qui suit la dernière conversation de l'extrait : il ne voit personne, personne ne le voit.
        var documents = batch.Documents!;
        var last = documents.max(1).values.unsqueeze(1).expand_as(documents);
        Assert.True(documents.masked_select(padding).equal(last.masked_select(padding)).all().item<bool>());
    }

    [Fact]
    public void OnlyRepliesAreScoredInsideExtracts()
    {
        // Le masque des extraits est celui du texte (réponses + leur <|end|>), aux places de y.
        using var dataset = Dataset();
        using var scope = NewDisposeScope();
        var batch = dataset.GetBatch(validation: false, 64, BlockSize, CPU);
        var y = batch.Y.data<long>().ToArray();
        var mask = batch.Mask.data<float>().ToArray();

        var tokenizer = TestData.Tokenizer;
        var neverScored = new[] { DocToken, UserToken, AssistantToken }.Select(t => (long)tokenizer.SpecialId(t)).ToHashSet();
        Assert.Contains(mask, m => m == 1);
        Assert.Contains(mask, m => m == 0);
        for (int i = 0; i < y.Length; i++)
            if (mask[i] == 1)
                Assert.DoesNotContain(y[i], neverScored);
    }

    [Fact]
    public void ValidationExtractsAreAlwaysTheSame()
    {
        // Même avec une autre graine de batchs : mêmes extraits de vérification, mêmes conversations.
        using var a = Dataset(seed: 42);
        using var b = Dataset(seed: 1042);
        using var scope = NewDisposeScope();
        var va = a.FixedValidation(16, BlockSize);
        var vb = b.FixedValidation(16, BlockSize);
        Assert.True(va.X.equal(vb.X).all().item<bool>());
        Assert.True(va.Mask.equal(vb.Mask).all().item<bool>());
        Assert.True(va.Documents!.equal(vb.Documents!).all().item<bool>());
    }

    [Fact]
    public void ExtractFillCountsOnlyRealText()
    {
        using var dataset = Dataset();
        double fill = dataset.ExtractFill(BlockSize);
        Assert.InRange(fill, 0.3, 0.999);

        using var plain = new TextDataset(Text, 42, CPU);
        Assert.Equal(1.0, plain.ExtractFill(BlockSize));

        // Mesuré sur des extraits tirés au hasard : proche de la moyenne exacte.
        using var scope = NewDisposeScope();
        double measured = 1 - Padding(dataset.GetBatch(validation: false, 2000, BlockSize, CPU)).to(ScalarType.Float32).mean().item<float>();
        Assert.InRange(measured, fill - 0.03, fill + 0.03);
    }

    [Fact]
    public void PlainTextHasNoDocuments()
    {
        using var plain = new TextDataset(TestData.RandomTokens(5_000), 42, CPU);
        using var scope = NewDisposeScope();
        Assert.Null(plain.GetBatch(validation: false, 4, 16, CPU).Documents);
    }
}
