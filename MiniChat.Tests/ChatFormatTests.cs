using MiniChat.Chat;
using MiniChat.Data;
using Xunit;
using static MiniChat.Tokenization.BpeTokenizer;

namespace MiniChat.Tests;

public sealed class ChatFormatTests
{
    [Fact]
    public void DialogueFileBecomesTheModelFormat()
    {
        string file = "Toi: Bonjour !\nIA: Bonjour,\ncomment vas-tu ?\n\nToi:Paris ?\nIA: La capitale.\n";

        Assert.Equal(
            DocToken + UserToken + "Bonjour !" + EndToken + AssistantToken + "Bonjour,\ncomment vas-tu ?" + EndToken +
            DocToken + UserToken + "Paris ?" + EndToken + AssistantToken + "La capitale." + EndToken,
            ChatFormat.FromDialogueFile(file));
    }

    [Fact]
    public void LinesBeforeTheFirstMessageAreIgnored() =>
        Assert.Equal(DocToken + UserToken + "Salut" + EndToken,
            ChatFormat.FromDialogueFile("fin d'une conversation coupée\nToi: Salut"));

    [Fact]
    public void TurnIsUserThenAssistant() =>
        Assert.Equal(UserToken + "Salut" + EndToken + AssistantToken + "Coucou" + EndToken, ChatFormat.Turn(" Salut ", "Coucou\n"));

    [Fact]
    public void ReplyWeightsCoverOnlyTheRepliesAndTheirEnd()
    {
        string formatted = UserToken + "Salut" + EndToken + AssistantToken + "Coucou" + EndToken;
        string expected = new string('0', (UserToken + "Salut" + EndToken + AssistantToken).Length) +
                          new string('1', ("Coucou" + EndToken).Length);

        Assert.Equal(expected, string.Concat(ChatFormat.ReplyWeights(formatted).Select(w => w.ToString())));
    }

    [Fact]
    public void TruncatedReplyCountsUntilTheEnd()
    {
        string formatted = UserToken + "Salut" + EndToken + AssistantToken + "Cou";
        var weights = ChatFormat.ReplyWeights(formatted);
        Assert.All(weights[^3..], w => Assert.Equal(1, w));
        Assert.All(weights[..^3], w => Assert.Equal(0, w));
    }

    [Fact]
    public void LossMaskKeepsOnlyRepliesAndTheirEndToken()
    {
        var tokenizer = TestData.Tokenizer;
        string text = ChatFormat.FromDialogueFile(TestData.Dialogues);
        var tokenized = TextDataset.Tokenize(text, tokenizer, onlyReplies: true);

        Assert.NotNull(tokenized.Mask);
        Assert.Equal(tokenized.Ids.Length, tokenized.Mask.Length);

        // Ce qui est noté (masque à 1), mis bout à bout : exactement les réponses, chacune avec son <|end|>.
        string scored = string.Concat(tokenized.Ids.Where((_, i) => tokenized.Mask[i] == 1).Select(id => tokenizer.Decode(id)));
        Assert.Equal("Bonjour, comment vas-tu ?" + EndToken + "Tant mieux !" + EndToken +
                     "Paris est la capitale de la France." + EndToken, scored);

        // Le reste (lu comme contexte, jamais noté) : les questions, leur <|end|>, <|doc|> et <|assistant|>.
        string context = string.Concat(tokenized.Ids.Where((_, i) => tokenized.Mask[i] == 0).Select(id => tokenizer.Decode(id)));
        Assert.Equal(DocToken + UserToken + "Bonjour !" + EndToken + AssistantToken + UserToken + "Très bien, merci." + EndToken +
                     AssistantToken + DocToken + UserToken + "Quelle est la capitale de la France ?" + EndToken + AssistantToken,
            context);
    }

    [Fact]
    public void PlainTextHasNoMask()
    {
        var tokenized = TextDataset.Tokenize(TestData.Text, TestData.Tokenizer, onlyReplies: false);
        Assert.Null(tokenized.Mask);
        Assert.Equal(TestData.Text, TestData.Tokenizer.Decode(tokenized.Ids.Select(id => (long)id)));
    }
}
