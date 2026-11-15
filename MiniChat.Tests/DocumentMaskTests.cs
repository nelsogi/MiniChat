using MiniChat.Config;
using MiniChat.Model;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace MiniChat.Tests;

/// <summary>Plusieurs conversations dans un extrait : chacune ne voit qu'elle-même, ses positions repartent de 0.</summary>
public sealed class DocumentMaskTests
{
    [Fact]
    public void ATokenSeesOnlyTheTokensBeforeItInItsOwnConversation()
    {
        using var scope = NewDisposeScope();
        var mask = DocumentMask.From(tensor(new long[] { 0, 0, 0, 1, 1, 2 }).unsqueeze(0));

        Assert.Equal(new long[] { 1, 1, 6, 6 }, mask.Attention.shape);
        bool[,] expected =
        {
            { true,  false, false, false, false, false },
            { true,  true,  false, false, false, false },
            { true,  true,  true,  false, false, false },
            { false, false, false, true,  false, false },   // la conversation 1 ne voit rien de la 0
            { false, false, false, true,  true,  false },
            { false, false, false, false, false, true  },   // la 2 ne voit qu'elle-même
        };
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                Assert.True(expected[i, j] == mask.Attention[0, 0, i, j].item<bool>(), $"ligne {i}, colonne {j}");
    }

    [Fact]
    public void PositionsRestartAtEachConversation()
    {
        using var scope = NewDisposeScope();
        var documents = tensor(new long[,] { { 0, 0, 0, 1, 1, 2, 2, 2 }, { 0, 0, 0, 0, 0, 0, 0, 0 } });

        var positions = DocumentMask.From(documents).Positions;

        Assert.Equal(new long[] { 0, 1, 2, 0, 1, 0, 1, 2 }, positions[0].data<long>().ToArray());
        Assert.Equal(new long[] { 0, 1, 2, 3, 4, 5, 6, 7 }, positions[1].data<long>().ToArray());   // une seule : 0 à 7
    }

    [Fact]
    public void EveryTokenSeesAtLeastItself()
    {
        // Une ligne vide donnerait NaN dans le softmax : la diagonale est toujours visible.
        using var scope = NewDisposeScope();
        var mask = DocumentMask.From(tensor(new long[] { 0, 1, 2, 3, 3, 4 }).unsqueeze(0));
        Assert.True(mask.Attention[0, 0].diagonal().all().item<bool>());
    }

    [Theory]
    [InlineData(ModelArchitecture.Modern)]
    [InlineData(ModelArchitecture.Classic)]
    public void PackedConversationsGiveTheSameLogitsAsEachAlone(ModelArchitecture architecture)
    {
        // Trois conversations mises bout à bout dans un extrait de 16 tokens = chacune passée seule dans le modèle :
        // le masque (rien ne passe d'une conversation à l'autre) et les positions (RoPE ou embedding) le garantissent.
        using var scope = NewDisposeScope();
        using var model = TestData.TinyModel(architecture: architecture);
        model.eval();
        using var noGrad = no_grad();

        long[][] conversations = [[5, 9, 2, 7, 7, 3], [11, 4, 8, 1, 6], [2, 2, 10, 12, 13]];
        var ids = tensor(conversations.SelectMany(c => c).ToArray()).unsqueeze(0);
        var documents = tensor(conversations.SelectMany((c, d) => Enumerable.Repeat((long)d, c.Length)).ToArray()).unsqueeze(0);

        var packed = model.Forward(ids, cache: null, documents);
        int start = 0;
        foreach (var conversation in conversations)
        {
            var alone = model.forward(tensor(conversation).unsqueeze(0));
            var part = packed.narrow(1, start, conversation.Length);
            Assert.True(part.allclose(alone, rtol: 1e-4, atol: 1e-5), $"conversation à la place {start}");
            start += conversation.Length;
        }
    }

    [Fact]
    public void WithoutDocumentsTheModelIsUnchanged()
    {
        // Un seul document partout = le masque causal simple : même résultat qu'avant (pré-entraînement).
        using var scope = NewDisposeScope();
        using var model = TestData.TinyModel();
        model.eval();
        using var noGrad = no_grad();

        var ids = tensor(new long[] { 5, 9, 2, 7, 11, 4, 8, 1 }).unsqueeze(0);
        var single = model.Forward(ids, cache: null, zeros_like(ids));
        Assert.True(single.allclose(model.forward(ids), rtol: 1e-4, atol: 1e-5));
    }

    [Fact]
    public void AConversationIsNotChangedByTheOneBeforeIt()
    {
        // Même deuxième conversation, première différente : la deuxième donne les mêmes logits.
        using var scope = NewDisposeScope();
        using var model = TestData.TinyModel();
        model.eval();
        using var noGrad = no_grad();
        var documents = tensor(new long[] { 0, 0, 0, 0, 1, 1, 1, 1 }).unsqueeze(0);

        var a = model.Forward(tensor(new long[] { 1, 2, 3, 4, 20, 21, 22, 23 }).unsqueeze(0), cache: null, documents);
        var b = model.Forward(tensor(new long[] { 9, 8, 7, 6, 20, 21, 22, 23 }).unsqueeze(0), cache: null, documents);

        Assert.True(a.narrow(1, 4, 4).allclose(b.narrow(1, 4, 4), rtol: 1e-4, atol: 1e-5));
        Assert.False(a.narrow(1, 0, 4).allclose(b.narrow(1, 0, 4), rtol: 1e-4, atol: 1e-5));
    }
}
