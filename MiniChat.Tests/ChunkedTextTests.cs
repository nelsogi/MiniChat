using MiniChat.Chat;
using MiniChat.Data;
using Xunit;
using static TorchSharp.torch;

namespace MiniChat.Tests;

/// <summary>
/// Les tokens en morceaux (ChunkedArray) : avec des morceaux de 1 000 tokens, beaucoup d'extraits sont à cheval
/// sur deux morceaux, et tout doit être exactement comme avec un seul morceau.
/// </summary>
public sealed class ChunkedTextTests
{
    private const int SmallChunk = 1_000;
    private const int BlockSize = 16;

    [Fact]
    public void ChunkedArrayReadsAcrossChunks()
    {
        var values = Enumerable.Range(0, 3_500).Select(i => (short)i).ToArray();
        var chunked = ChunkedArray<short>.From(values, SmallChunk);

        Assert.Equal(3_500, chunked.Length);
        Assert.Equal([1_000, 1_000, 1_000, 500], chunked.Chunks.Select(c => c.Length));
        Assert.Equal(values, chunked);
        Assert.Equal((short)2_999, chunked[2_999]);

        // 17 tokens de 990 à 1 006 (deux morceaux), et 2 100 tokens de 900 à 2 999 (trois morceaux).
        var across = new short[17];
        chunked.CopyTo(990, across);
        Assert.Equal(values[990..1_007], across);
        var three = new short[2_100];
        chunked.CopyTo(900, three);
        Assert.Equal(values[900..3_000], three);
        Assert.Throws<ArgumentOutOfRangeException>(() => chunked.CopyTo(3_490, new short[11]));
    }

    [Fact]
    public void BuilderMakesFullChunksAndAShortLastOne()
    {
        var builder = new ChunkedArray<byte>.Builder(SmallChunk);
        for (int i = 0; i < 2_001; i++)
            builder.Add((byte)i);
        var built = builder.Build();

        Assert.Equal(2_001, built.Length);
        Assert.Equal([1_000, 1_000, 1], built.Chunks.Select(c => c.Length));
        Assert.Empty(new ChunkedArray<byte>.Builder(SmallChunk).Build());
    }

    [Fact]
    public void TextExtractsAreTheSameWithSmallChunks()
    {
        var random = new Random(7);
        var ids = Enumerable.Range(0, 60_000).Select(_ => (short)random.Next(TestData.Tokenizer.VocabSize)).ToArray();
        using var whole = new TextDataset(new TokenizedText(ids, null, ids.Length), seed: 3, CPU);
        using var chunked = new TextDataset(new TokenizedText(ids, null, ids.Length, SmallChunk), seed: 3, CPU);

        // Même découpage entraînement/vérification, mêmes extraits tirés, mêmes extraits de vérification. Avec
        // 2 000 extraits de 17 tokens et des morceaux de 1 000, une cinquantaine sont à cheval sur deux morceaux.
        Assert.Equal(whole.TrainLength, chunked.TrainLength);
        Assert.Equal(whole.ValidationLength, chunked.ValidationLength);
        for (int i = 0; i < 250; i++)
            AssertSame(whole.GetBatch(i % 5 == 0, 8, BlockSize, CPU), chunked.GetBatch(i % 5 == 0, 8, BlockSize, CPU));
        AssertSame(whole.FixedValidation(40, BlockSize), chunked.FixedValidation(40, BlockSize));
    }

    [Fact]
    public void ExtractsAreTheIdsAtTheirPlace()
    {
        // Token n° i = i % 30 000 : chaque extrait doit être une suite de nombres qui se suivent, même à cheval.
        var ids = Enumerable.Range(0, 30_000).Select(i => (short)i).ToArray();
        using var dataset = new TextDataset(new TokenizedText(ids, null, ids.Length, SmallChunk), seed: 1, CPU);
        int across = 0;
        for (int i = 0; i < 300; i++)
        {
            var batch = dataset.GetBatch(false, 8, BlockSize, CPU);
            var x = batch.X.data<long>().ToArray();
            var y = batch.Y.data<long>().ToArray();
            for (int b = 0; b < 8; b++)
            {
                long start = x[b * BlockSize];
                Assert.Equal(Enumerable.Range(0, BlockSize).Select(t => start + t), x.Skip(b * BlockSize).Take(BlockSize));
                Assert.Equal(Enumerable.Range(1, BlockSize).Select(t => start + t), y.Skip(b * BlockSize).Take(BlockSize));
                if (start / SmallChunk != (start + BlockSize) / SmallChunk)
                    across++;
            }
        }
        Assert.True(across > 10, $"seulement {across} extraits à cheval sur deux morceaux");
    }

    [Fact]
    public void DialogueExtractsAreTheSameWithSmallChunks()
    {
        string formatted = ChatFormat.FromDialogueFile(TestData.DialogueFile(400));
        var whole = TextDataset.Tokenize(formatted, TestData.Tokenizer, onlyReplies: true);
        var small = TextDataset.Tokenize(formatted, TestData.Tokenizer, onlyReplies: true, SmallChunk);
        int doc = TestData.Tokenizer.SpecialId(MiniChat.Tokenization.BpeTokenizer.DocToken);

        Assert.True(small.Ids.Chunks.Count > 3);
        Assert.Equal(whole.Ids, small.Ids);
        Assert.Equal(whole.Mask!, small.Mask!);

        var (train, _) = TextDataset.Split(whole.Ids.Length, 10);
        Assert.Equal(TextDataset.Conversations(whole.Ids, train, (short)doc), TextDataset.Conversations(small.Ids, train, (short)doc));

        using var a = new TextDataset(whole, seed: 5, CPU, documentToken: doc);
        using var b = new TextDataset(small, seed: 5, CPU, documentToken: doc);
        for (int i = 0; i < 100; i++)
            AssertSame(a.GetBatch(i % 4 == 0, 4, 32, CPU), b.GetBatch(i % 4 == 0, 4, 32, CPU));
        AssertSame(a.FixedValidation(20, 32), b.FixedValidation(20, 32));
    }

    [Fact]
    public void FileTokenizedInSmallChunksIsTheSame()
    {
        using var folder = new TempFolder();
        string path = folder.File("texte.txt");
        File.WriteAllText(path, string.Concat(Enumerable.Repeat(TestData.Text, 40)));

        var whole = TextDataset.TokenizeFile(path, TestData.Tokenizer);
        var small = TextDataset.TokenizeFile(path, TestData.Tokenizer, SmallChunk);

        Assert.True(small.Ids.Chunks.Count > 3);
        Assert.Equal(whole.Ids, small.Ids);
        Assert.Equal(whole.Characters, small.Characters);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CacheWrittenInSmallChunksIsReadBackTheSame(bool onlyReplies)
    {
        using var folder = new TempFolder();
        string source = folder.File("source.txt");
        File.WriteAllText(source, onlyReplies ? TestData.DialogueFile(200) : string.Concat(Enumerable.Repeat(TestData.Text, 40)));
        TokenizedText Tokenize() => onlyReplies
            ? TextDataset.Tokenize(ChatFormat.FromDialogueFile(File.ReadAllText(source)), TestData.Tokenizer, true, SmallChunk)
            : TextDataset.TokenizeFile(source, TestData.Tokenizer, SmallChunk);

        var written = TokenCache.Load(folder.File("cache"), "texte", [source], "", TestData.Tokenizer, onlyReplies, Tokenize, SmallChunk);
        var smallChunks = TokenCache.Load(folder.File("cache"), "texte", [source], "", TestData.Tokenizer, onlyReplies,
            () => throw new InvalidOperationException("pas relu"), SmallChunk);
        var oneChunk = TokenCache.Load(folder.File("cache"), "texte", [source], "", TestData.Tokenizer, onlyReplies,
            () => throw new InvalidOperationException("pas relu"));

        Assert.False(written.FromCache);
        Assert.True(smallChunks.FromCache);
        Assert.True(written.Text.Ids.Chunks.Count > 3);
        Assert.Equal(written.Text.Ids.Chunks.Select(c => c.Length), smallChunks.Text.Ids.Chunks.Select(c => c.Length));
        Assert.Single(oneChunk.Text.Ids.Chunks);   // le fichier ne dépend pas de la taille des morceaux
        foreach (var read in new[] { smallChunks.Text, oneChunk.Text })
        {
            Assert.Equal(written.Text.Ids, read.Ids);
            Assert.Equal(written.Text.Characters, read.Characters);
            if (onlyReplies) Assert.Equal(written.Text.Mask!, read.Mask!);
            else Assert.Null(read.Mask);
        }
    }

    [Theory]
    [InlineData(1_500)]
    [InlineData(35_421)]
    [InlineData(1_000_003)]
    [InlineData(23_877_929)]
    public void SplitIsTheSameAsWithIntIndexes(int length)
    {
        var (train, validation) = TextDataset.Split(length, 10);
        var (oldTrain, oldValidation) = IntSplit(length, 10);

        Assert.Equal(oldTrain, train.Select(s => ((int)s.Start, (int)s.Length)));
        Assert.Equal(oldValidation, validation.Select(s => ((int)s.Start, (int)s.Length)));
    }

    [Fact]
    public void SplitWorksBeyondTwoBillionTokens()
    {
        // Au-delà de int.MaxValue : seulement le découpage (pas de tokens en mémoire), en long de bout en bout.
        long length = 5_000_000_000;
        var (train, validation) = TextDataset.Split(length, 10);
        var all = train.Concat(validation).OrderBy(s => s.Start).ToList();

        Assert.Equal(0, all[0].Start);
        for (int i = 1; i < all.Count; i++)
            Assert.Equal(all[i - 1].Start + all[i - 1].Length, all[i].Start);
        Assert.Equal(length, all[^1].Start + all[^1].Length);
        Assert.InRange(validation.Sum(s => s.Length) / (double)length, 0.09, 0.11);
    }

    // TextDataset.Split d'avant les morceaux, avec des index en int : le découpage ne doit pas avoir changé.
    private static (List<(int, int)> Train, List<(int, int)> Validation) IntSplit(int length, int validationEvery)
    {
        int chunk = Math.Max(1, Math.Max(Math.Min(length / 100, TextDataset.ChunkTokens), Math.Min(1_000, length / validationEvery)));
        int chunks = (length + chunk - 1) / chunk;
        var random = new Random(1234);
        var isValidation = new bool[chunks];
        for (int group = 0; group < chunks; group += validationEvery)
        {
            int size = Math.Min(validationEvery, chunks - group);
            if (size == validationEvery || group == 0 || random.Next(validationEvery) < size)
                isValidation[group + random.Next(size)] = true;
        }
        var train = new List<(int, int)>();
        var validation = new List<(int, int)>();
        int first = 0;
        for (int i = 1; i <= chunks; i++)
        {
            if (i < chunks && isValidation[i] == isValidation[first])
                continue;
            int start = first * chunk;
            (isValidation[first] ? validation : train).Add((start, Math.Min(i * chunk, length) - start));
            first = i;
        }
        return (train, validation);
    }

    private static void AssertSame(Batch expected, Batch actual)
    {
        Assert.True(expected.X.equal(actual.X).all().item<bool>(), "X");
        Assert.True(expected.Y.equal(actual.Y).all().item<bool>(), "Y");
        Assert.True(expected.Mask.equal(actual.Mask).all().item<bool>(), "Mask");
        Assert.Equal(expected.Documents is null, actual.Documents is null);
        if (expected.Documents is not null)
            Assert.True(expected.Documents.equal(actual.Documents!).all().item<bool>(), "Documents");
    }
}
