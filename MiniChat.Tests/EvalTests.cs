using MiniChat.Evaluation;
using MiniChat.Generation;
using Xunit;
using static TorchSharp.torch;

namespace MiniChat.Tests;

public sealed class EvalTests
{
    // data/eval.json du projet (copié à côté des tests par le csproj).
    private static readonly string EvalPath = Path.Combine(AppContext.BaseDirectory, "eval.json");

    [Theory]
    [InlineData("La capitale de la France est Paris.", "Paris")]
    [InlineData("PARIS", "Paris")]
    [InlineData("C'est Pékin.", "Pékin")]
    [InlineData("C'est pekin.", "Pékin")]                  // sans accent dans la réponse
    [InlineData("Elle est à Athenes", "Athènes")]
    [InlineData("Le coeur.", "cœur")]                      // œ = oe
    [InlineData("La formule est H2O.", "H₂O")]
    [InlineData("Marie Antoinette", "Marie-Antoinette")]   // ponctuation = espace
    [InlineData("Jeanne d’Arc, en 1431.", "Jeanne d'Arc")]   // apostrophe typographique
    [InlineData("2 + 2 = 4", "4")]
    public void ReplyContainingAnAcceptedAnswerIsCorrect(string reply, string answer) =>
        Assert.True(EvalTest.IsCorrect(reply, [answer]));

    [Theory]
    [InlineData("Cela fait 14.", "4")]                     // en mots entiers : 4 n'est pas dans 14
    [InlineData("Il est excellent.", "lent")]
    [InlineData("Lyon", "Paris")]
    [InlineData("", "Paris")]
    public void OtherRepliesAreWrong(string reply, string answer) =>
        Assert.False(EvalTest.IsCorrect(reply, [answer]));

    [Fact]
    public void AnyAcceptedAnswerCounts() =>
        Assert.True(EvalTest.IsCorrect("Beijing, bien sûr.", ["Pékin", "Beijing"]));

    [Fact]
    public void EvalFileHasThreeHundredQuestionsInFiveCategories()
    {
        var questions = EvalTest.ReadQuestions(EvalPath);

        Assert.Equal(300, questions.Count);
        var categories = questions.GroupBy(q => q.Category).ToDictionary(g => g.Key, g => g.Count());
        Assert.Equal(["capitales", "sciences", "histoire", "vocabulaire", "calcul"], categories.Keys);
        Assert.All(categories.Values, count => Assert.Equal(60, count));
        Assert.Equal(questions.Count, questions.Select(q => EvalTest.Normalize(q.Question)).Distinct().Count());
        // Chaque réponse acceptée garde au moins un mot une fois simplifiée.
        Assert.All(questions.SelectMany(q => q.Answers), a => Assert.NotEmpty(EvalTest.Normalize(a)));
    }

    [Fact]
    public void NoQuestionContainsItsOwnAnswer()
    {
        // Sinon un modèle qui répète la question aurait juste.
        Assert.Empty(EvalTest.ReadQuestions(EvalPath).Where(q => EvalTest.IsCorrect(q.Question, q.Answers)).Select(q => q.Question));
    }

    [Fact]
    public void NoEvalQuestionIsInTheDialogues()
    {
        // Les dialogues du fine-tuning (Data/ du projet, pas copiés à côté des tests) : une question de /eval qui y est
        // mot pour mot (sans tenir compte des majuscules, des accents ni de la ponctuation) mesurerait la mémoire du
        // modèle, pas ce qu'il sait. DataBuilder sans-eval les retire. dialogues.txt n'est pas dans git : sans lui
        // (dépôt tout juste cloné), rien à vérifier.
        var questions = EvalTest.ReadQuestions(EvalPath).Select(q => $" {EvalTest.Normalize(q.Question)} ").ToArray();
        foreach (string path in new[] { "dialogues.txt", "identite.txt" }.Select(TestData.ProjectDataFile).Where(File.Exists))
        {
            var conversations = File.ReadAllText(path).Replace("\r\n", "\n").Split("\n\n");
            var found = new System.Collections.Concurrent.ConcurrentBag<string>();
            Parallel.ForEach(conversations, conversation =>
            {
                string text = $" {EvalTest.Normalize(conversation)} ";
                foreach (string question in questions.Where(q => text.Contains(q, StringComparison.Ordinal)))
                    found.Add(question.Trim());
            });
            Assert.True(found.IsEmpty, $"{Path.GetFileName(path)} contient des questions de eval.json : " +
                                       string.Join(" ; ", found.Distinct().Take(10)));
        }
    }

    [Fact]
    public void ScoresAreCountedPerCategory()
    {
        EvalQuestion Question(string category) => new(category, "?", ["x"]);
        var result = new EvalResult(
        [
            new EvalAnswer(Question("a"), "x", true), new EvalAnswer(Question("a"), "y", false),
            new EvalAnswer(Question("b"), "x", true),
        ], QuestionCount: 4);

        Assert.Equal([new EvalScore("a", 1, 2), new EvalScore("b", 1, 1)], result.Scores);
        Assert.Equal((2, 3), (result.Correct, result.Total));
        Assert.False(result.Complete);
    }

    [Fact]
    public void GreedyAnswersAreTheSameEveryTime()
    {
        using var model = TestData.TinyModel();
        var generator = new TextGenerator(model, TestData.Tokenizer, CPU);
        var questions = EvalTest.ReadQuestions(EvalPath).Take(3).ToList();

        var first = EvalTest.Run(generator, questions);
        manual_seed(999);   // le hasard global n'y change rien : toujours le token le plus probable
        var second = EvalTest.Run(generator, questions);

        Assert.True(first.Complete);
        Assert.Equal(first.Answers.Select(a => a.Reply), second.Answers.Select(a => a.Reply));
    }

    [Fact]
    public void HistoryKeepsEveryEvaluationOfAModel()
    {
        using var folder = new TempFolder();
        var history = new EvalHistory(folder.Path);
        var scores = new List<EvalScore> { new("capitales", 12, 20) };

        history.Add(new EvalRecord(new DateTime(2026, 11, 1), "model", "run1", 100, 3.2f, 12, 20, scores));
        history.Add(new EvalRecord(new DateTime(2026, 11, 2), "model", "run2", 200, 3.0f, 15, 20, scores));
        history.Add(new EvalRecord(new DateTime(2026, 11, 2), "autre", null, null, null, 1, 20, scores));

        var records = history.Read("model");
        Assert.Equal(["run1", "run2"], records.Select(r => r.RunId));
        Assert.Equal(15, records[^1].Correct);
        Assert.Equal(scores, records[^1].Scores);
        Assert.Single(history.Read("autre"));
        Assert.Empty(history.Read("inconnu"));
    }
}
