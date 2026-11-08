using System.Text;
using MiniChat.Chat;
using MiniChat.Generation;

namespace MiniChat.Evaluation;

/// <summary>
/// Le même petit examen après chaque entraînement : les questions de data/questions-test.txt,
/// toujours avec la même température et la même graine. Seul le modèle change d'un entraînement
/// à l'autre, pas le hasard : on peut comparer les fichiers results/*.txt entre eux.
/// </summary>
public static class QuestionTest
{
    public const double Temperature = 0.7;
    public const long Seed = 1234;

    /// <summary>Une question par ligne. Les lignes vides et celles qui commencent par # sont ignorées.</summary>
    public static string[] ReadQuestions(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Fichier de questions introuvable : {path}");

        var questions = File.ReadAllLines(path)
            .Select(line => line.Trim())
            .Where(line => line.Length > 0 && !line.StartsWith('#'))
            .ToArray();

        return questions.Length > 0
            ? questions
            : throw new InvalidDataException($"Aucune question dans {path}");
    }

    /// <summary>
    /// Réponse à une question, sans historique de conversation, avec les réglages fixes.
    /// Chaque question repart de la même graine : sa réponse ne dépend pas des questions d'avant.
    /// </summary>
    public static string Answer(TextGenerator generator, string question)
    {
        var chat = new ChatSession(generator) { Temperature = Temperature, Seed = Seed };
        return chat.Send(question, _ => { });
    }

    /// <summary>Pendant l'entraînement : une réponse d'exemple, pour voir le modèle progresser.</summary>
    public static string Sample(TextGenerator generator, string question, int step) =>
        $"Exemple (étape {step}) :\n" +
        $"  {ChatSession.UserPrefix}{question}\n" +
        $"  {ChatSession.BotPrefix}{Answer(generator, question)}";

    /// <summary>
    /// Pose toutes les questions, écrit les réponses dans outputPath et renvoie le rapport (pour l'afficher).
    /// cancellation : s'arrête avant la question suivante, le rapport garde les réponses déjà données.
    /// </summary>
    public static string Run(TextGenerator generator, string questionsPath, string outputPath, string description,
        CancellationToken cancellation = default)
    {
        var defaults = new ChatSession(generator);
        var report = new StringBuilder();
        report.AppendLine($"MiniChat - questions de test - {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        report.AppendLine(description);
        report.AppendLine($"Réglages : température {Temperature}, top-k {defaults.TopK}, top-p {defaults.TopP}, graine {Seed}");
        report.AppendLine();

        var questions = ReadQuestions(questionsPath);
        for (int i = 0; i < questions.Length; i++)
        {
            if (cancellation.IsCancellationRequested)
            {
                report.AppendLine($"(interrompu : {questions.Length - i} question(s) sans réponse)");
                break;
            }
            report.AppendLine($"{i + 1,2}. {ChatSession.UserPrefix}{questions[i]}");
            report.AppendLine($"    {ChatSession.BotPrefix}{Answer(generator, questions[i])}");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllText(outputPath, report.ToString(), new UTF8Encoding(false));
        return report.ToString().TrimEnd();
    }
}
