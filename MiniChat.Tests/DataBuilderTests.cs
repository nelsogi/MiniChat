using DataBuilder;
using Parquet.Serialization;
using Xunit;

namespace MiniChat.Tests;

public sealed class OasstConverterTests
{
    // Un arbre OpenAssistant : deux réponses à la question de départ, deux relances sous la meilleure, et une
    // relance dont la meilleure réponse contient du code (inutilisable). Plus un arbre d'un seul tour.
    private static readonly (string, string?, string, string, int?)[] Messages =
    [
        ("p1", null, "prompter", "Bonjour, comment vas-tu ?", null),
        ("a1", "p1", "assistant", "Très bien, merci.", 0),
        ("a1b", "p1", "assistant", "Bien.", 1),
        ("p2", "a1", "prompter", "Que fais-tu ?", 0),
        ("a2", "p2", "assistant", "Je réponds à tes questions.", 0),
        ("p3", "a1", "prompter", "Tu aimes lire ?", 1),
        ("a3", "p3", "assistant", "```code```", 0),
        ("a3b", "p3", "assistant", "Oui, beaucoup.", 1),
        ("p4", "a1b", "prompter", "Pourquoi seulement bien ?", 0),
        ("a4", "p4", "assistant", "Parce que.", 0),
        ("q1", null, "prompter", "Quelle heure est-il ?", null),
        ("r1", "q1", "assistant", "Je ne sais pas.", 0),
    ];

    [Fact]
    public void EveryPathToALeafUnderTheBestRepliesIsKept()
    {
        var converter = new OasstConverter("fr", 300, 2);
        string output = converter.Convert(Messages);

        Assert.Equal(
            "Toi: Bonjour, comment vas-tu ?\nIA: Très bien, merci.\nToi: Que fais-tu ?\nIA: Je réponds à tes questions.\n\n" +
            "Toi: Bonjour, comment vas-tu ?\nIA: Très bien, merci.\nToi: Tu aimes lire ?\nIA: Oui, beaucoup.\n",
            output);
        Assert.Equal(2, converter.Kept);
        Assert.Equal(1, converter.Rejected["moins de 2 tours"]);   // l'arbre d'un seul tour
    }

    [Fact]
    public void TurnsAreCountedBeforeAndAfter()
    {
        var converter = new OasstConverter("fr", 300, 2);
        converter.Convert(Messages);

        // Avant : une conversation par arbre (2 tours en suivant toujours le mieux classé, et l'arbre d'un tour).
        Assert.Equal(new SortedDictionary<int, int> { [1] = 1, [2] = 1 }, converter.TurnsBefore);
        Assert.Equal(new SortedDictionary<int, int> { [2] = 2 }, converter.TurnsAfter);
    }

    [Fact]
    public void MinTurnsCanKeepSingleTurnConversations()
    {
        var converter = new OasstConverter("fr", 300, 2, minTurns: 1);
        string output = converter.Convert(Messages);

        Assert.Equal(3, converter.Kept);
        Assert.EndsWith("Toi: Quelle heure est-il ?\nIA: Je ne sais pas.\n", output);
        Assert.Equal(new SortedDictionary<int, int> { [1] = 1, [2] = 2 }, converter.TurnsAfter);
    }

    [Fact]
    public void UnusableFollowUpStopsThePathAtTheReply()
    {
        (string, string?, string, string, int?)[] messages =
        [
            ("p1", null, "prompter", "Bonjour !", null),
            ("a1", "p1", "assistant", "Bonjour, que puis-je faire pour toi ?", 0),
            ("p2", "a1", "prompter", "Écris du code : ```print(1)```", 0),
            ("a2", "p2", "assistant", "Voilà.", 0),
        ];
        var converter = new OasstConverter("fr", 300, 2, minTurns: 1);
        string output = converter.Convert(messages);

        Assert.Equal("Toi: Bonjour !\nIA: Bonjour, que puis-je faire pour toi ?\n", output);
        Assert.Equal(1, converter.Shortened);
    }
}

public sealed class CalculGeneratorTests
{
    private static readonly string EvalPath = Path.Combine(AppContext.BaseDirectory, "eval.json");

    private static List<(string Question, string Reply, CalculGenerator.Problem Problem)> Generate(int count, int seed) =>
        new CalculGenerator(CalculGenerator.ReadEvalQuestions(EvalPath)).Generate(count, seed);

    [Fact]
    public void SameSeedGivesTheSameDialogues()
    {
        Assert.Equal(CalculGenerator.Format(Generate(500, 1)), CalculGenerator.Format(Generate(500, 1)));
        Assert.NotEqual(CalculGenerator.Format(Generate(500, 1)), CalculGenerator.Format(Generate(500, 2)));
    }

    [Fact]
    public void TwentyThousandCorrectDialoguesWithoutEvalQuestions()
    {
        var evalQuestions = CalculGenerator.ReadEvalQuestions(EvalPath).ToList();
        var evalSimplified = evalQuestions.Select(CalculGenerator.Simplify).ToHashSet();
        var evalProblems = evalQuestions.SelectMany(CalculGenerator.ProblemsIn).ToHashSet();
        var dialogues = Generate(20_000, 1);

        Assert.Equal(20_000, dialogues.Count);
        Assert.Equal(dialogues.Count, dialogues.Select(d => d.Question).Distinct().Count());
        Assert.All(Enum.GetValues<CalculGenerator.Operation>(),
            op => Assert.InRange(dialogues.Count(d => d.Problem.Op == op), 6_666, 6_667));
        Assert.All(dialogues, d =>
        {
            Assert.InRange(d.Problem.A, 0, CalculGenerator.MaxNumber);
            Assert.InRange(d.Problem.B, 0, CalculGenerator.MaxNumber);
            Assert.True(d.Problem.Result >= 0, d.Question);
            Assert.Contains($" {d.Problem.Result} ", $" {CalculGenerator.Simplify(d.Reply)} ");
            Assert.DoesNotContain(CalculGenerator.Simplify(d.Question), evalSimplified);
            Assert.DoesNotContain(d.Problem, evalProblems);
            // La question demande bien ce calcul (aussi pour les petits problèmes : « 12 boîtes de 3 crayons »).
            Assert.Contains(d.Problem.A.ToString(), d.Question);
            Assert.Contains(d.Problem.B.ToString(), d.Question);
        });
    }

    [Fact]
    public void FormatIsTheDialogueFile()
    {
        string text = CalculGenerator.Format(Generate(2, 1));
        var conversations = text.Split("\n\n");

        Assert.Equal(2, conversations.Length);
        Assert.All(conversations, c => Assert.Matches(@"^Toi: [^\n]+\nIA: [^\n]+\n?$", c));
        Assert.EndsWith("\n", text);
    }

    [Theory]
    [InlineData("Combien font 7 + 5 ?", CalculGenerator.Operation.Addition, 5, 7)]       // 7 + 5 = 5 + 7
    [InlineData("Combien font 10 - 3 ?", CalculGenerator.Operation.Subtraction, 10, 3)]
    [InlineData("Combien font 3 fois 4 ?", CalculGenerator.Operation.Multiplication, 3, 4)]
    [InlineData("Combien font 6 x 4 ?", CalculGenerator.Operation.Multiplication, 4, 6)]
    [InlineData("Combien fait 9 plus 8 ?", CalculGenerator.Operation.Addition, 8, 9)]
    [InlineData("Quel est le double de 16 ?", CalculGenerator.Operation.Multiplication, 2, 16)]
    public void CalculationsAreFoundInQuestions(string question, CalculGenerator.Operation op, int a, int b) =>
        Assert.Equal([new CalculGenerator.Problem(op, a, b)], CalculGenerator.ProblemsIn(question));

    [Fact]
    public void EvalCalculationIsNeverAskedInAnyWording()
    {
        var generator = new CalculGenerator(["Combien font 7 + 5 ?", "Combien font 3 fois 4 ?"]);
        var problems = generator.Generate(15_000, 3).Select(d => d.Problem).ToHashSet();

        Assert.DoesNotContain(CalculGenerator.Problem.Of(CalculGenerator.Operation.Addition, 7, 5), problems);
        Assert.DoesNotContain(CalculGenerator.Problem.Of(CalculGenerator.Operation.Multiplication, 4, 3), problems);
        Assert.True(problems.Count > 5_000);   // les autres calculs, oui
    }
}

public sealed class EvalFilterTests
{
    [Fact]
    public void ConversationsWithAnEvalQuestionAreRemoved()
    {
        const string dialogues =
            "Toi: Bonjour !\nIA: Bonjour.\n\n" +
            "Toi: Dis-moi, quelle est la capitale de la france?\nIA: Paris.\n\n" +     // majuscules, ponctuation : trouvée
            "Toi: J'aime la France.\nIA: Moi aussi.\n\n" +
            "Toi: Salut\nIA: Salut !\nToi: Quelle est la capitale de la France ?\nIA: Paris.\n";   // au 2e tour
        var filter = new EvalFilter(["Quelle est la capitale de la France ?"]);

        string kept = filter.Filter(dialogues);

        Assert.Equal("Toi: Bonjour !\nIA: Bonjour.\n\nToi: J'aime la France.\nIA: Moi aussi.\n", kept);
        Assert.Equal(2, filter.Removed);
        Assert.Equal(2, filter.Kept);
        Assert.Equal(2, filter.Found["quelle est la capitale de la france"]);
    }

    [Fact]
    public void OnlyWholeWordsMatch()
    {
        var filter = new EvalFilter(["Combien font 2 + 2 ?"]);
        string kept = filter.Filter("Toi: Combien font 2 + 25 ?\nIA: 27.\n");

        Assert.Equal(0, filter.Removed);
        Assert.Equal("Toi: Combien font 2 + 25 ?\nIA: 27.\n", kept);
    }
}

public sealed class ParquetConverterTests
{
    private sealed class Row
    {
        public string text { get; set; } = "";
    }

    private static async Task<string> WriteParquet(TempFolder folder, string name, params string[] documents)
    {
        string path = folder.File(name);
        await using var file = File.Create(path);
        await ParquetSerializer.SerializeAsync(documents.Select(d => new Row { text = d }), file);
        return path;
    }

    [Fact]
    public async Task DuplicatesAreWrittenOnce()
    {
        using var folder = new TempFolder();
        string input = await WriteParquet(folder, "a.parquet", "Premier article.", "Deuxième article.", "Premier article.");
        string output = folder.File("texte.txt");

        var converter = new ParquetConverter("text", null);
        await converter.ConvertAsync([input], output);

        Assert.Equal(["Premier article.\n", "Deuxième article.\n"], ParquetConverter.ReadDocuments(output));
        Assert.Equal(2, converter.Kept);
        Assert.Equal(1, converter.Duplicates);
    }

    [Fact]
    public async Task AppendKeepsTheFileAndSkipsItsDocuments()
    {
        using var folder = new TempFolder();
        string output = folder.File("texte.txt");
        await new ParquetConverter("text", null).ConvertAsync([await WriteParquet(folder, "a.parquet", "Un.", "Deux.")], output);

        var converter = new ParquetConverter("text", null);
        await converter.ConvertAsync([await WriteParquet(folder, "b.parquet", "Deux.", "Trois.")], output, append: true);

        Assert.Equal("<|doc|>Un.\n<|doc|>Deux.\n<|doc|>Trois.\n", File.ReadAllText(output));
        Assert.Equal(2, converter.Known);
        Assert.Equal(1, converter.Kept);
        Assert.Equal(1, converter.Duplicates);
    }

    [Fact]
    public async Task ExcludedFilesAreNotTakenAgain()
    {
        using var folder = new TempFolder();
        string used = await WriteParquet(folder, "deja.parquet", "Ancien article.\nSur deux lignes.", "Autre ancien.");
        string input = await WriteParquet(folder, "nouveau.parquet", "Nouvel article.", "Ancien article.\nSur deux lignes.");
        string output = folder.File("texte.txt");

        var converter = new ParquetConverter("text", null);
        await converter.ConvertAsync([input], output, exclude: [used]);

        Assert.Equal("<|doc|>Nouvel article.\n", File.ReadAllText(output));
        Assert.Equal(1, converter.Duplicates);
    }

    [Fact]
    public void DocumentsAreReadBackAsWritten()
    {
        using var folder = new TempFolder();
        string path = folder.File("texte.txt");
        File.WriteAllText(path, "Avant le premier.\n<|doc|>Un.\nDeux lignes.\n<|doc|>Fin<|doc|>Collé.\n");

        Assert.Equal(["Avant le premier.\n", "Un.\nDeux lignes.\n", "Fin", "Collé.\n"], ParquetConverter.ReadDocuments(path));
    }
}
