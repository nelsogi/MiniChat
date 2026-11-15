using MiniChat.Data;
using Xunit;
using static TorchSharp.torch;

namespace MiniChat.Tests;

public sealed class TextDatasetSplitTests
{
    // Gros texte (morceaux de 10 000 tokens), dialogues (~35 000 tokens), tout petit texte, longueur quelconque.
    public static TheoryData<int> Lengths => [1_000_000, 35_000, 5_000, 1_234_567, 98_765];

    [Theory]
    [MemberData(nameof(Lengths))]
    public void TrainAndValidationCoverTheTextWithoutOverlap(int length)
    {
        var (train, validation) = TextDataset.Split(length, validationEvery: 10);

        // Mis bout à bout dans l'ordre, les segments des deux côtés couvrent [0, length) exactement une fois.
        var all = train.Select(s => (s.Start, s.Length, Validation: false))
            .Concat(validation.Select(s => (s.Start, s.Length, Validation: true)))
            .OrderBy(s => s.Start)
            .ToList();
        long position = 0;
        foreach (var segment in all)
        {
            Assert.Equal(position, segment.Start);
            Assert.True(segment.Length > 0);
            position += segment.Length;
        }
        Assert.Equal(length, position);

        // Deux segments voisins sont toujours de côtés différents (les morceaux voisins du même côté sont regroupés).
        for (int i = 1; i < all.Count; i++)
            Assert.NotEqual(all[i - 1].Validation, all[i].Validation);
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void AboutTenPercentGoesToValidation(int length)
    {
        var (_, validation) = TextDataset.Split(length, validationEvery: 10);
        double share = (double)validation.Sum(s => s.Length) / length;
        Assert.InRange(share, 0.07, 0.13);
    }

    [Fact]
    public void ValidationIsSpreadOverTheWholeText()
    {
        // Un morceau sur dix dans chaque groupe de dix : de la vérification dans chaque dixième du texte.
        const int length = 1_000_000;
        var (_, validation) = TextDataset.Split(length, validationEvery: 10);
        for (int tenth = 0; tenth < 10; tenth++)
        {
            var (first, last) = (tenth * length / 10, (tenth + 1) * length / 10);
            Assert.Equal(length / 100, validation.Sum(s => Math.Max(0, Math.Min(s.Start + s.Length, last) - Math.Max(s.Start, first))));
        }
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void SplitIsTheSameEveryTime(int length)
    {
        var first = TextDataset.Split(length, validationEvery: 10);
        var second = TextDataset.Split(length, validationEvery: 10);
        Assert.Equal(first.Train, second.Train);
        Assert.Equal(first.Validation, second.Validation);
    }

    [Fact]
    public void SplitDoesNotDependOnTheBatchSeed()
    {
        // La graine des batchs change à la reprise et entre les phases : ni le découpage ni les extraits
        // de vérification ne doivent changer avec elle.
        var text = TestData.RandomTokens(20_000);
        using var a = new TextDataset(text, seed: 42, CPU);
        using var b = new TextDataset(text, seed: 1042, CPU);

        Assert.Equal(a.TrainLength, b.TrainLength);
        Assert.Equal(a.ValidationLength, b.ValidationLength);
        Assert.Equal(text.Ids.Length, a.TrainLength + a.ValidationLength);

        var (xa, ya, _, _) = a.FixedValidation(8, 16);
        var (xb, yb, _, _) = b.FixedValidation(8, 16);
        Assert.True(xa.equal(xb).all().item<bool>());
        Assert.True(ya.equal(yb).all().item<bool>());
    }
}
