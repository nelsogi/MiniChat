using MiniChat.Chat;
using MiniChat.Data;
using MiniChat.Tokenization;
using Xunit;

namespace MiniChat.Tests;

public sealed class TokenCacheTests : IDisposable
{
    private readonly TempFolder _folder = new();
    private readonly string _cache;
    private readonly string _source;
    private int _tokenized;   // combien de fois le texte a vraiment été découpé

    public TokenCacheTests()
    {
        _cache = _folder.File("cache");
        _source = _folder.File("texte.txt");
        File.WriteAllText(_source, TestData.Text);
    }

    public void Dispose() => _folder.Dispose();

    private (TokenizedText Text, bool FromCache) Load(BpeTokenizer tokenizer, string settings = "", bool onlyReplies = false) =>
        TokenCache.Load(_cache, "texte", [_source], settings, tokenizer, onlyReplies, () =>
        {
            _tokenized++;
            return onlyReplies
                ? TextDataset.Tokenize(ChatFormat.FromDialogueFile(File.ReadAllText(_source)), tokenizer, onlyReplies: true)
                : TextDataset.TokenizeFile(_source, tokenizer);
        });

    [Fact]
    public void SecondLoadComesFromTheCacheWithTheSameTokens()
    {
        var first = Load(TestData.Tokenizer);
        var second = Load(TestData.Tokenizer);

        Assert.False(first.FromCache);
        Assert.True(second.FromCache);
        Assert.Equal(1, _tokenized);
        Assert.Equal(first.Text.Ids, second.Text.Ids);
        Assert.Equal(first.Text.Characters, second.Text.Characters);
        Assert.Null(second.Text.Mask);
    }

    [Fact]
    public void ChangedFileIsTokenizedAgain()
    {
        var before = Load(TestData.Tokenizer);
        File.AppendAllText(_source, "Le chat joue.\n");

        var after = Load(TestData.Tokenizer);

        Assert.False(after.FromCache);
        Assert.Equal(2, _tokenized);
        Assert.NotEqual(before.Text.Ids, after.Text.Ids);
        Assert.True(Load(TestData.Tokenizer).FromCache);   // puis gardé à son tour
    }

    [Fact]
    public void FileWithTheSameSizeButOtherContentIsTokenizedAgain()
    {
        Load(TestData.Tokenizer);
        File.WriteAllText(_source, TestData.Text.Replace("chat", "chou"));

        Assert.False(Load(TestData.Tokenizer).FromCache);
    }

    [Fact]
    public void OtherTokenizerIsTokenizedAgain()
    {
        var other = BpeTokenizer.Train(TestData.Text, 80);
        var first = Load(TestData.Tokenizer);

        var withOther = Load(other);

        Assert.False(withOther.FromCache);
        Assert.Equal(2, _tokenized);
        Assert.Equal(TestData.Text, other.Decode(withOther.Text.Ids.Select(id => (long)id)).Replace("\r\n", "\n"));
        Assert.NotEqual(first.Text.Ids, withOther.Text.Ids);
    }

    [Fact]
    public void OtherSettingsOrMaskAreTokenizedAgain()
    {
        Load(TestData.Tokenizer);
        Assert.False(Load(TestData.Tokenizer, settings: "identite x10").FromCache);
        Assert.False(Load(TestData.Tokenizer, settings: "identite x10", onlyReplies: true).FromCache);
        Assert.True(Load(TestData.Tokenizer, settings: "identite x10", onlyReplies: true).FromCache);
    }

    [Fact]
    public void MaskIsKeptInTheCache()
    {
        File.WriteAllText(_source, TestData.Dialogues);
        var first = Load(TestData.Tokenizer, onlyReplies: true);
        var second = Load(TestData.Tokenizer, onlyReplies: true);

        Assert.True(second.FromCache);
        Assert.Equal(first.Text.Mask, second.Text.Mask);
    }

    [Fact]
    public void OnlyOneCacheFilePerName()
    {
        Load(TestData.Tokenizer);
        File.AppendAllText(_source, "Le chien dort.\n");
        Load(TestData.Tokenizer);

        Assert.Single(Directory.GetFiles(_cache, "texte-*.bin"));
    }

    [Fact]
    public void TruncatedCacheIsTokenizedAgain()
    {
        Load(TestData.Tokenizer);
        string file = Directory.GetFiles(_cache, "texte-*.bin").Single();
        File.WriteAllBytes(file, File.ReadAllBytes(file)[..100]);

        Assert.False(Load(TestData.Tokenizer).FromCache);
    }
}
