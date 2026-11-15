using MiniChat.Chat;
using MiniChat.Tokenization;
using Xunit;
using static MiniChat.Tokenization.BpeTokenizer;

namespace MiniChat.Tests;

public sealed class BpeTokenizerTests
{
    private static BpeTokenizer Tokenizer => TestData.Tokenizer;

    [Theory]
    [InlineData("Le chat dort sur le canapé.")]
    [InlineData("Bonjour ! Comment vas-tu ?\nTrès bien, merci.")]
    [InlineData("  Les   enfants\n\nvont à l'école à 17 h ; Paris, Lyon...")]
    [InlineData("chien chat chien canapé jardin balle")]   // des mots appris, dans un autre ordre
    [InlineData("")]
    public void EncodeThenDecodeGivesBackTheText(string text) =>
        Assert.Equal(text, Tokenizer.Decode(Tokenizer.Encode(text)));

    [Fact]
    public void EncodeThenDecodeGivesBackAConversation()
    {
        string conversation = DocToken + ChatFormat.Turn("Bonjour !", "Bonjour, comment vas-tu ?") + ChatFormat.User("Très bien.");
        Assert.Equal(conversation, Tokenizer.Decode(Tokenizer.Encode(conversation)));
    }

    [Fact]
    public void FrequentWordsBecomeFewerTokens() =>
        Assert.True(Tokenizer.Encode(" chat").Length < " chat".Length);

    [Theory]
    [InlineData(DocToken)]
    [InlineData(UserToken)]
    [InlineData(AssistantToken)]
    [InlineData(EndToken)]
    public void SpecialTokenIsOneToken(string special) =>
        Assert.Equal([Tokenizer.SpecialId(special)], Tokenizer.Encode(special));

    [Fact]
    public void SpecialTokensAreNeverSplitEvenGluedToWords()
    {
        long[] ids = Tokenizer.Encode("chat" + EndToken + "chien" + UserToken + AssistantToken + "!");

        Assert.Equal(3, ids.Count(id => Tokenizer.SpecialIds.Contains((int)id)));
        Assert.Equal(Tokenizer.Encode("chat"), ids[..Array.IndexOf(ids, (long)Tokenizer.SpecialId(EndToken))]);
        Assert.Equal(Tokenizer.SpecialId(UserToken), ids[^3]);
        Assert.Equal(Tokenizer.SpecialId(AssistantToken), ids[^2]);
    }

    [Fact]
    public void NoOrdinaryTokenContainsASpecialToken()
    {
        // Les tokens spéciaux du texte d'entraînement ne prêtent pas leurs caractères aux fusions.
        var ordinary = Enumerable.Range(0, Tokenizer.VocabSize).Where(id => !Tokenizer.SpecialIds.Contains(id));
        Assert.All(ordinary, id => Assert.DoesNotContain("<|", Tokenizer.Decode(id)));
        Assert.Equal(DefaultSpecialTokens, Tokenizer.SpecialTokens);
    }

    [Fact]
    public void SpecialTokenPositionIsItsLastCharacter()
    {
        string text = "chat" + EndToken;
        var last = Tokenizer.EncodeWithPositions(text).Last();
        Assert.Equal((Tokenizer.SpecialId(EndToken), text.Length - 1), last);
    }

    [Fact]
    public void UnknownCharactersAreIgnored() =>
        Assert.Equal("chat", Tokenizer.Decode(Tokenizer.Encode("ch☃at")));

    [Fact]
    public void TrainingIsDeterministic()
    {
        var again = Train(TestData.Text + "\n" + ChatFormat.FromDialogueFile(TestData.Dialogues), 120);
        Assert.Equal(Tokenizer.Alphabet, again.Alphabet);
        Assert.Equal(Tokenizer.Merges, again.Merges);
    }

    [Fact]
    public void VocabularyHasTheRequestedSize() => Assert.Equal(120, Tokenizer.VocabSize);
}
