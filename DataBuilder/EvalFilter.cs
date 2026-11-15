namespace DataBuilder;

/// <summary>
/// Retire d'un dialogues.txt les conversations qui contiennent une question de eval.json mot pour mot (sans tenir
/// compte des majuscules, des accents ni de la ponctuation, en mots entiers : voir CalculGenerator.Simplify).
/// Sinon /eval mesurerait en partie ce que le modèle a appris par cœur : « Quelle est la capitale de la France ? »
/// est 110 fois dans les dialogues de Luth-SFT et des autres jeux.
/// </summary>
public sealed class EvalFilter
{
    private readonly string[] _questions;

    /// <summary>Conversations retirées par question (dans le texte simplifié de la question).</summary>
    public Dictionary<string, int> Found { get; } = new();

    public int Kept { get; private set; }
    public int Removed { get; private set; }

    public EvalFilter(IEnumerable<string> evalQuestions)
    {
        _questions = evalQuestions.Select(CalculGenerator.Simplify).Where(q => q.Length > 0).Distinct().ToArray();
    }

    /// <summary>Les conversations d'un fichier Toi:/IA: (une ligne vide entre deux), sans celles qui contiennent une question.</summary>
    public string Filter(string dialogues)
    {
        var conversations = dialogues.Replace("\r\n", "\n").Split("\n\n", StringSplitOptions.RemoveEmptyEntries)
            .Select(c => c.Trim('\n')).Where(c => c.Length > 0).ToArray();

        // ~300 000 conversations x 300 questions : en parallèle, puis dans l'ordre du fichier.
        var matches = new string?[conversations.Length];
        Parallel.For(0, conversations.Length, i =>
        {
            string text = $" {CalculGenerator.Simplify(conversations[i])} ";
            matches[i] = _questions.FirstOrDefault(q => text.Contains($" {q} ", StringComparison.Ordinal));
        });

        var kept = new List<string>(conversations.Length);
        for (int i = 0; i < conversations.Length; i++)
        {
            if (matches[i] is { } question)
            {
                Found[question] = Found.GetValueOrDefault(question) + 1;
                Removed++;
            }
            else
                kept.Add(conversations[i]);
        }
        Kept = kept.Count;
        return kept.Count == 0 ? "" : string.Join("\n\n", kept) + "\n";
    }
}
